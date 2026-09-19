"""Audit existing trainable UMI windows without changing labels or window indices."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import build_umi_window_index as indexer
from umi_window_rules import evaluate_rows, evaluate_edges
from umi_quality_metrics import summarize, merge, finalize

WORK = {}
ROW_SCOPES = ("raw", "row_valid", "window_covered")
EDGE_SCOPES = ("raw_available", "row_valid_contiguous", "window_internal")
CASE_SCHEMA = pa.schema([
    ("case_id",pa.string()),("episode_index",pa.int64()),("data_file",pa.string()),
    ("file_row_offset",pa.int64()),("episode_row_offset",pa.int64()),("frame_index",pa.int64()),
    ("timestamp",pa.float64()),("source_timestamp_ns",pa.int64()),("task_index",pa.int64()),
    ("source_set_id",pa.string()),("source_mcap",pa.string()),("metric",pa.string()),
    ("scope",pa.string()),("selection",pa.string()),("value",pa.float64()),
    ("row_valid",pa.bool_()),("window_covered",pa.bool_()),("window_internal_edge",pa.bool_()),
    ("shortest_rotation_deg",pa.float64()),("negative_dot",pa.bool_()),
    ("timestamp_delta",pa.float64()),("source_delta_seconds",pa.float64()),
    ("context_json",pa.string()),
])
THRESHOLDS = {"position_norm":(5.,10.,50.),"gripper":(.06,.10,.20),
              "rotation_deg":(15.,30.,90.)}
PRIORITY = {"data/chunk-000/file-599.parquet", "data/chunk-001/file-166.parquet",
            "data/chunk-001/file-191.parquet", "data/chunk-000/file-157.parquet",
            "data/chunk-003/file-179.parquet"}


def clean_scalar(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def clean_vector(x):
    return [clean_scalar(v) for v in x]


def init_worker(source, output, window_root, tasks, fps, index_fingerprint, audit_fingerprint):
    WORK.update(source=Path(source),output=Path(output),windows=Path(window_root),
                tasks=set(tasks),fps=fps,index_fingerprint=index_fingerprint,
                fingerprint=audit_fingerprint)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    signal.signal(signal.SIGINT,signal.SIG_IGN)


def source_values_fp64(table):
    """Read original stored numeric components independently of validity masking."""
    n=len(table)
    output=np.full((n,16),np.nan,dtype=np.float64)
    offset=0
    for key,width in zip(indexer.SIGNAL_COLUMNS,indexer.SIGNAL_WIDTHS):
        if key not in table.column_names:
            offset+=width
            continue
        arr=table[key].combine_chunks()
        if pa.types.is_floating(arr.type) or pa.types.is_integer(arr.type):
            if width==1:
                output[:,offset]=arr.to_numpy(zero_copy_only=False).astype(np.float64)
        elif pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
            starts=arr.offsets.to_numpy(zero_copy_only=False)
            sizes=np.diff(starts)
            good=arr.is_valid().to_numpy(zero_copy_only=False) & (sizes==width)
            rows=np.flatnonzero(good)
            child=arr.values
            if len(rows) and (pa.types.is_floating(child.type) or pa.types.is_integer(child.type)):
                take=(starts[rows,None]+np.arange(width)).reshape(-1)
                raw=pc.take(child,pa.array(take,type=pa.int64())).to_numpy(zero_copy_only=False)
                output[rows,offset:offset+width]=raw.astype(np.float64).reshape(-1,width)
        else:
            raise ValueError("Unsupported source numeric type for raw audit: "+key+" "+str(arr.type))
        offset+=width
    return output


def audit_file(job):
    relative = job["data_file"]
    ident = indexer.unit_id(relative)
    root, output = WORK["windows"],WORK["output"]
    old_marker = indexer.verified_completed(job,root,WORK["source"],WORK["index_fingerprint"])
    if old_marker is None:
        raise ValueError("No completed original index unit")
    path = WORK["source"]/relative
    before = indexer.signature(path)
    pqfile = pq.ParquetFile(path)
    columns = indexer.selected_columns(pqfile.schema_arrow)
    if pqfile.metadata.num_rows > 1000000:
        raise ValueError("File exceeds one-million-row memory guard")
    table = pqfile.read(columns=columns,use_threads=False)
    evaluated = evaluate_rows(table,WORK["tasks"])
    original_values = source_values_fp64(table)
    parts = pq.read_table(root/"episode_segments"/(ident+".parquet")).to_pylist()
    ranges = pq.read_table(root/"valid_anchor_ranges"/(ident+".parquet")).to_pylist()
    qualities = pq.read_table(root/"episode_quality_parts"/(ident+".parquet")).to_pylist()
    quality_by_ep = {x["episode_index"]:x for x in qualities}
    specs = {x["episode_index"]:x for x in job["episodes"]}
    by_ep, by_range = {},{}
    for seg in parts:
        by_ep.setdefault(seg["episode_index"],[]).append(seg)
    for span in ranges:
        by_range.setdefault(span["episode_index"],[]).append(span)
    metrics,events,counts,best = {},{},Counter(),{}
    def record_metric(key,values,kind):
        incoming = summarize(values,kind)
        metrics[key] = merge(metrics.get(key),incoming)
    def record_event(key,mask,source):
        number = int(mask.sum())
        if not number:
            return
        record = events.setdefault(key,{"count":0,"episodes":0,"by_source_set":{}})
        record["count"] += number
        record["episodes"] += 1
        group = record["by_source_set"].setdefault(source,{"count":0,"episodes":0})
        group["count"] += number
        group["episodes"] += 1
    for ep,segments in by_ep.items():
        segments.sort(key=lambda x:x["episode_row_offset_start"])
        offset=0
        arrays=[]
        for seg in segments:
            if seg["episode_row_offset_start"] != offset:
                raise ValueError("Episode segment offsets are not consecutive")
            a,b=seg["file_row_start"],seg["file_row_end_exclusive"]
            if a<0 or b>len(table) or b<=a:
                raise ValueError("Invalid stored file-row segment")
            arrays.append(np.arange(a,b,dtype=np.int64))
            offset += b-a
        physical = np.concatenate(arrays)
        ev = indexer.subset_evaluation(evaluated,physical)
        n=len(physical)
        if n != quality_by_ep[ep]["raw_rows"] or not np.all(ev["episode"]==ep):
            raise ValueError("Stored episode location no longer matches source")
        valid_frame=ev["frame_valid"]
        _,inv,freq=np.unique(ev["frame"][valid_frame],return_inverse=True,return_counts=True)
        duplicate=np.zeros(n,dtype=bool)
        duplicate[valid_frame]=freq[inv]>1
        row_ok=ev["row_ok"] & ~duplicate
        edges=evaluate_edges(ev,WORK["fps"])
        edge_ok=row_ok[:-1] & row_ok[1:]
        for mask in edges.values():
            edge_ok &= ~mask
        covered=np.zeros(n,dtype=bool)
        internal=np.zeros(max(n-1,0),dtype=bool)
        span_count=0
        for span in by_range.get(ep,[]):
            a,b=span["anchor_start"],span["anchor_end_exclusive"]
            if a<0 or b<=a or b+16>n:
                raise ValueError("Stored anchor range invalid")
            covered[a:b+16]=True
            internal[a:b+15]=True
            span_count += b-a
        qold=quality_by_ep[ep]
        if (row_ok.sum()!=qold["valid_rows"] or covered.sum()!=qold["window_covered_rows"]
                or span_count!=qold["valid_windows"]):
            raise ValueError("Existing index scope/count disagrees with source rules")
        if np.any(covered & ~row_ok) or np.any(internal & ~edge_ok):
            raise ValueError("Stored trainable windows include invalid rows/edges")
        masks={"raw":np.ones(n,dtype=bool),"row_valid":row_ok,"window_covered":covered}
        edge_masks={"raw_available":np.ones(max(n-1,0),dtype=bool),
                    "row_valid_contiguous":edge_ok,"window_internal":internal}
        counts.update(episodes=1,raw_rows=n,row_valid_rows=int(row_ok.sum()),
                      window_covered_rows=int(covered.sum()),window_internal_edges=int(internal.sum()),
                      row_valid_contiguous_edges=int(edge_ok.sum()))
        values=original_values[physical]
        source=specs[ep].get("source_set_id") or "<missing>"
        angles_by_hand={}
        negatives_by_hand={}
        def consider(metric,scope,selection,series,mask,smallest=False,angles=None,negative=None):
            candidates=np.flatnonzero(mask & np.isfinite(series))
            if not len(candidates):
                return
            chosen=int(candidates[np.argmin(series[candidates]) if smallest else np.argmax(series[candidates])])
            val=float(series[chosen])
            key=metric+"|"+scope+"|"+selection
            previous=best.get(key)
            if previous is not None and ((smallest and val>=previous["value"]) or (not smallest and val<=previous["value"])):
                return
            context=[]
            for p in range(max(0,chosen-4),min(n,chosen+6)):
                context.append({"episode_row_offset":p,"file_row_offset":int(physical[p]),
                    "frame_index":int(ev["frame"][p]),"timestamp":clean_scalar(ev["timestamp"][p]),
                    "source_timestamp_ns":int(ev["root_ns"][p]) if ev["root_ns"] is not None else None,
                    "row_valid":bool(row_ok[p]),"window_covered":bool(covered[p]),
                    "world_pose_action_16d":clean_vector(values[p])})
            is_edge=chosen<n-1
            ns=ev["root_ns"]
            best[key]={"case_id":f"{ep}:{chosen}:{key}","episode_index":ep,"data_file":relative,
                "file_row_offset":int(physical[chosen]),"episode_row_offset":chosen,
                "frame_index":int(ev["frame"][chosen]),"timestamp":clean_scalar(ev["timestamp"][chosen]),
                "source_timestamp_ns":int(ns[chosen]) if ns is not None else None,
                "task_index":int(ev["task"][chosen]),"source_set_id":specs[ep].get("source_set_id"),
                "source_mcap":specs[ep].get("source_mcap"),"metric":metric,"scope":scope,
                "selection":selection,"value":val,"row_valid":bool(row_ok[chosen]),
                "window_covered":bool(covered[chosen]),"window_internal_edge":bool(internal[chosen]) if is_edge else False,
                "shortest_rotation_deg":clean_scalar(angles[chosen]) if angles is not None and is_edge else None,
                "negative_dot":bool(negative[chosen]) if negative is not None and is_edge else None,
                "timestamp_delta":clean_scalar(ev["timestamp"][chosen+1]-ev["timestamp"][chosen]) if is_edge else None,
                "source_delta_seconds":(int(ns[chosen+1])-int(ns[chosen]))/1e9 if ns is not None and is_edge else None,
                "context_json":indexer.packed(context)}
        for off,hand in ((0,"robot1"),(8,"robot2")):
            xyz=values[:,off:off+3]
            quat=values[:,off+3:off+7]
            norms=np.linalg.norm(quat,axis=1)
            position=np.linalg.norm(xyz,axis=1)
            gripper=values[:,off+7]
            for kind,series in (("position_norm",position),("gripper",gripper),("quaternion_norm",norms)):
                for scope,mask in masks.items():
                    record_metric(hand+"."+kind+"."+scope,series[mask],kind)
                    for threshold in THRESHOLDS.get(kind,()):
                        record_event(hand+"."+kind+"."+scope+".gt_"+str(threshold),
                                     mask & np.isfinite(series) & (series>threshold),source)
                if kind in ("position_norm","gripper"):
                    for scope in ("raw","window_covered"):
                        consider(hand+"."+kind,scope,"maximum",series,masks[scope])
            good=(np.isfinite(norms[:-1]) & np.isfinite(norms[1:]) &
                  (norms[:-1]>0) & (norms[1:]>0))
            dot=np.full(max(n-1,0),np.nan,dtype=np.float64)
            dot[good]=np.sum((quat[:-1][good]/norms[:-1][good,None])*
                             (quat[1:][good]/norms[1:][good,None]),axis=1)
            angle=np.degrees(2*np.arccos(np.clip(abs(dot),0,1)))
            negative=dot<0
            translation=np.linalg.norm(xyz[1:]-xyz[:-1],axis=1)
            angles_by_hand[hand]=angle
            negatives_by_hand[hand]=negative
            for scope,mask in edge_masks.items():
                record_metric(hand+".rotation_deg."+scope,angle[mask],"rotation_deg")
                record_metric(hand+".translation."+scope,translation[mask],"translation")
                record_metric(hand+".negative_dot_rotation_deg."+scope,angle[mask & negative],
                              "negative_dot_rotation_deg")
                record_event(hand+".negative_dot."+scope,mask & negative,source)
                for threshold in THRESHOLDS["rotation_deg"]:
                    record_event(hand+".rotation_deg."+scope+".gt_"+str(threshold),
                                 mask & np.isfinite(angle) & (angle>threshold),source)
            for scope in ("row_valid_contiguous","window_internal"):
                consider(hand+".rotation_deg",scope,"maximum",angle,edge_masks[scope],angles=angle,negative=negative)
            consider(hand+".negative_dot_rotation_deg","window_internal","minimum_rotation_with_negative_dot",
                     angle,internal & negative,smallest=True,angles=angle,negative=negative)
            consider(hand+".negative_dot_rotation_deg","window_internal","maximum_rotation_with_negative_dot",
                     angle,internal & negative,angles=angle,negative=negative)
        # One ordinary candidate per file, from the first episode with an internal
        # edge and small rotation on both hands. This is a comparison, not random sampling.
        if "ordinary" not in best:
            ordinary=internal.copy()
            for angle in angles_by_hand.values():
                ordinary &= np.isfinite(angle) & (angle<5)
            choices=np.flatnonzero(ordinary)
            if len(choices):
                p=int(choices[len(choices)//2])
                mask=np.zeros(n,dtype=bool); mask[p]=True
                consider("ordinary","window_covered","comparison",np.zeros(n),mask)
                best["ordinary"]=best.pop("ordinary|window_covered|comparison")
    if indexer.signature(path)!=before:
        raise ValueError("Source changed during audit")
    case_path=output/"case_parts"/(ident+".parquet")
    indexer.atomic_parquet(case_path,list(best.values()),CASE_SCHEMA)
    result={"status":"completed","data_file":relative,"source_identity":before,
        "audit_fingerprint":WORK["fingerprint"],"counts":dict(counts),
        "metrics":metrics,"events":events,"case_part":str(case_path.relative_to(output)),
        "case_sha256":indexer.digest(case_path),"updated_at":indexer.now()}
    indexer.atomic_json(output/"completed"/(ident+".json"),result)
    return result


def merged_events(target,incoming):
    for key,value in incoming.items():
        t=target.setdefault(key,{"count":0,"episodes":0,"by_source_set":{}})
        t["count"]+=value["count"];t["episodes"]+=value["episodes"]
        for source,counts in value["by_source_set"].items():
            s=t["by_source_set"].setdefault(source,{"count":0,"episodes":0})
            s["count"]+=counts["count"];s["episodes"]+=counts["episodes"]


def publish(output,markers,expected_files,index_report,complete,failed):
    counts=Counter(); metrics={};events={}
    candidates={}
    for marker in markers.values():
        counts.update(marker["counts"])
        merged_events(events,marker["events"])
        for key,s in marker["metrics"].items():
            metrics[key]=merge(metrics.get(key),s)
        for case in pq.read_table(output/marker["case_part"]).to_pylist():
            key=(case["metric"],case["scope"],case["selection"])
            group=candidates.setdefault(key,[])
            group.append(case)
            group.sort(key=lambda x:x["value"],reverse=not key[2].startswith("minimum"))
            del group[6:]
    cases=[c for g in candidates.values() for c in g]
    indexer.atomic_parquet(output/"quality_cases.parquet",cases,CASE_SCHEMA)
    if complete:
        for key,oldkey in (("raw_rows","raw_rows"),("row_valid_rows","valid_rows"),
                           ("window_covered_rows","window_covered_rows")):
            if counts[key]!=index_report["counts"][oldkey]:
                raise ValueError("Full audit/index totals disagree: "+key)
    report={"status":"completed" if complete else "incomplete","scan_complete":complete,
        "scope":"Diagnostics only; no label/window/normalization changes",
        "numeric_precision":"FP64 diagnostics from original stored Arrow components; existing raw-FP32 rules only determine masks",
        "completed_files":len(markers),"expected_files":expected_files,"failed_files":failed,
        "counts":dict(counts),"metrics":{k:finalize(v) for k,v in metrics.items()},
        "diagnostic_events":events,"quality_case_count":len(cases),
        "video_content_validated":False,
        "source_group_basis":"catalog source_set_id; not scene/session/device identities",
        "quantiles":"Fixed histogram brackets, not exact empirical quantiles",
        "case_selection":"Per-file extrema and ordinary controls, then global top six per category; not a random prevalence estimate",
        "updated_at":indexer.now()}
    indexer.atomic_json(output/"trainable_quality_report.json",report)
    lines=["# UMI 训练范围质量复核","",f"状态：{report['status']}；完成 {len(markers)}/{expected_files} 个文件。",
           "","本轮不修改原始标签、窗口索引或模型。诊断桶不作为删除阈值。",
           "统计分别覆盖原始可解析行、行级有效行、窗口覆盖行；相邻边另按窗口内部范围计算。",
           "均值与方差使用 FP64 count/mean/M2 合并。分位数只报告直方图区间。",
           "","## 定位案例","", "| 指标 | 范围 | episode | 行位置 | 数值 |","|---|---|---:|---:|---:|"]
    for key,group in sorted(candidates.items()):
        if group:
            c=group[0]
            lines.append(f"| {key[0]} | {key[1]} | {c['episode_index']} | {c['episode_row_offset']} | {c['value']:.8g} |")
    lines += ["","## 待判断的含义","",
        "- 大世界系位置：检查持续偏置、轨迹变化及 Ego 同时刻位姿，不能由范数直接判坏。",
        "- 大开度：核对设备/转换语义与画面，不能直接裁剪或改单位。",
        "- 大最短旋转：已消除 q/-q 歧义，仍需检查真实动作与定位输出。",
        "- 负点积：结合对应最短旋转角，决定是否需要一致的表示变换。",
        "- 视频仅能通过后续有记录的定向查看确认；本统计本身没有解码视频。",
        "- 新增删帧/断边规则需创建新的窗口版本；本轮没有冻结新的清洗规则。",""]
    (output/"quality_review.md").write_text("\n".join(lines),encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",type=Path,default=indexer.PRIVATE/"roban_umi_world_pose_v1")
    parser.add_argument("--windows",type=Path,default=indexer.PRIVATE/"roban_umi_world_pose_windows_v1")
    parser.add_argument("--output",type=Path,default=indexer.PRIVATE/"roban_umi_trainable_quality_v1")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    output,windows=args.output.resolve(),args.windows.resolve()
    if args.workers<1: parser.error("workers must be positive")
    if not output.is_relative_to(indexer.PRIVATE.resolve()) or output==indexer.PRIVATE.resolve():
        parser.error("Output must be a private data_preparation subdirectory")
    for protected in (windows,args.catalog.resolve()):
        if output==protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            parser.error("Output must be separate from existing index/catalog")
    old=json.loads((windows/"window_index_config.json").read_text())
    index_report=json.loads((windows/"window_index_report.json").read_text())
    if not index_report.get("scan_complete") or index_report["status"]!="completed":
        raise ValueError("Require completed low-dimensional index")
    print("Reading existing index and metadata; no window regeneration",flush=True)
    source,info,tasks,jobs,contract,index_fingerprint=indexer.load_plan(args.catalog.resolve())
    if index_fingerprint!=old["rule_fingerprint"]:
        raise ValueError("Current source/rules/catalog do not match frozen window version")
    jobs_by_ep={e["episode_index"]:e for j in jobs for e in j["episodes"]}
    db_path=args.catalog.resolve()/"catalog_index.sqlite3"
    with sqlite3.connect(db_path.as_uri()+"?mode=ro",uri=True) as db:
        for ep,source_set,source_path in db.execute("SELECT episode_id,source_set,source_key FROM episodes"):
            jobs_by_ep[ep].update(source_set_id=source_set,source_mcap=source_path)
    jobs.sort(key=lambda j:(j["data_file"] not in PRIORITY,j["data_file"]))
    scripts=[Path(__file__),Path(__file__).with_name("umi_quality_metrics.py")]
    audit_contract={"index_fingerprint":index_fingerprint,"index_report_sha256":indexer.digest(windows/"window_index_report.json"),
        "code":{p.name:indexer.digest(p) for p in scripts},"windows":str(windows),"source":str(source),
        "row_scopes":ROW_SCOPES,"edge_scopes":EDGE_SCOPES,"diagnostic_thresholds":THRESHOLDS}
    fingerprint=hashlib.sha256(indexer.packed(audit_contract).encode()).hexdigest()
    if output.exists() and not args.resume: parser.error("Output exists; use --resume or new version")
    if not output.exists() and args.resume: parser.error("Resume output missing")
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/"audit.lock").open("a")
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: parser.error("Audit already running")
    cfg=output/"audit_config.json"
    if args.resume:
        if json.loads(cfg.read_text())["audit_fingerprint"]!=fingerprint: raise ValueError("Audit version changed")
    else:
        indexer.atomic_json(cfg,{"audit_fingerprint":fingerprint,"contract":audit_contract,
                                "created_at":indexer.now(),"expected_files":len(jobs)})
    for folder in ("completed","case_parts","failed"):
        (output/folder).mkdir(exist_ok=True)
    markers={};pending=[]
    for job in jobs:
        p=output/"completed"/(indexer.unit_id(job["data_file"])+".json")
        if p.exists():
            m=json.loads(p.read_text())
            if (m["audit_fingerprint"]!=fingerprint or m["source_identity"]!=indexer.signature(source/job["data_file"])
                    or indexer.digest(output/m["case_part"])!=m["case_sha256"]):
                raise ValueError("Completed audit unit changed/corrupt")
            markers[job["data_file"]]=m
        else: pending.append(job)
    failed={};stop={"value":False};start=time.monotonic()
    def handle(signum,frame): stop["value"]=True
    signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
    def progress():
        c=Counter()
        for m in markers.values(): c.update(m["counts"])
        state={"status":"stopping" if stop["value"] else "running","pid":os.getpid(),
            "completed_files":len(markers),"total_files":len(jobs),"failed_files":len(failed),
            "counts":dict(c),"updated_at":indexer.now(),"invocation_seconds":time.monotonic()-start}
        indexer.atomic_json(output/"progress.json",state)
        return state
    progress()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker,initargs=(str(source),str(output),str(windows),sorted(tasks),info["fps"],
                                             index_fingerprint,fingerprint)) as pool:
        it=iter(pending); active={}
        def submit():
            if stop["value"]: return
            j=next(it,None)
            if j is not None: active[pool.submit(audit_file,j)]=j
        for _ in range(args.workers): submit()
        last=time.monotonic()
        first_published=False
        while active:
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                j=active.pop(future)
                try: markers[j["data_file"]]=future.result()
                except Exception as error:
                    record={"data_file":j["data_file"],"error":str(error),"traceback":traceback.format_exc()}
                    failed[j["data_file"]]=record
                    indexer.atomic_json(output/"failed"/(indexer.unit_id(j["data_file"])+".json"),record)
                    print("FAILED",j["data_file"],str(error),flush=True)
                submit()
            if done or time.monotonic()-last>30:
                p=progress()
                if time.monotonic()-last>30 or len(markers)%25==0:
                    print(f"files={len(markers)}/{len(jobs)} failed={len(failed)} raw_rows={p['counts'].get('raw_rows',0)}",flush=True)
                    last=time.monotonic()
            if len(markers)>=8 and not first_published:
                publish(output,markers,len(jobs),index_report,False,list(failed.values()))
                first_published=True
                print("Preliminary cases published; full-scan statistics still incomplete",flush=True)
    complete=len(markers)==len(jobs) and not failed
    report=publish(output,markers,len(jobs),index_report,complete,list(failed.values()))
    p=progress();p["status"]=report["status"];indexer.atomic_json(output/"progress.json",p)
    print(indexer.packed({"status":report["status"],"counts":report["counts"]}),flush=True)
    if not complete: raise SystemExit(2)


if __name__=="__main__": main()
