"""Descarga, normaliza y versiona fuentes oficiales sin datos personales."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
import pandas as pd
from .config import SETTINGS as S
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def canonical_version(records):
    raw=json.dumps(records,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()
def ingest(verify=False):
    sources=json.loads((S.root/"references/sources.json").read_text())
    raw=S.root/"data/raw";raw.mkdir(parents=True,exist_ok=True)
    records=[];frames=[];seen={};duplicate=0;conflict=0;invalid=0;input_rows=0
    for source in sources:
        path=raw/source["name"]
        if not path.exists():
            if verify: raise FileNotFoundError(path)
            req=urllib.request.Request(source["url"],headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=120) as response,path.open("wb") as out:
                import shutil;shutil.copyfileobj(response,out)
        rec={**source,"sha256":sha256(path),"bytes":path.stat().st_size}
        if source["name"]=="estaciones.csv": records.append(rec);continue
        for encoding in ["utf-8-sig","cp1252"]:
            try:
                with path.open(encoding=encoding,errors="strict") as f:
                    for _ in f: pass
                break
            except UnicodeDecodeError: continue
        rec["encoding"]=encoding;rec["input_rows"]=0
        cols=["Viaje_Id","Inicio_del_viaje","Fin_del_viaje","Origen_Id","Destino_Id"]
        for chunk in pd.read_csv(path,encoding=encoding,usecols=cols,dtype="string",chunksize=S.chunk_size):
            rec["input_rows"]+=len(chunk);input_rows+=len(chunk)
            chunk.columns=["trip_id","started_at","ended_at","origin_id","destination_id"]
            for col in ["started_at","ended_at"]:chunk[col]=pd.to_datetime(chunk[col],format="%Y-%m-%d %H:%M:%S",errors="coerce")
            good=chunk.started_at.notna() & chunk.ended_at.notna() & (chunk.ended_at>=chunk.started_at) & chunk.origin_id.notna() & chunk.destination_id.notna() & chunk.trip_id.notna()
            invalid+=int((~good).sum());chunk=chunk[good]
            keep=[]
            for row in chunk.itertuples(index=False,name=None):
                key=row[0];fingerprint=tuple(row[1:])
                if key not in seen:seen[key]=fingerprint;keep.append(True)
                else:
                    keep.append(False)
                    if seen[key]==fingerprint:duplicate+=1
                    else:conflict+=1
            frames.append(chunk.loc[keep])
        records.append(rec)
    if conflict: raise ValueError(f"{conflict} trip IDs conflict; quarantine and investigate before training")
    trips=pd.concat(frames,ignore_index=True)
    trips=trips[(trips.started_at>=S.start)&(trips.started_at<S.end)]
    stations=pd.read_csv(raw/"estaciones.csv",dtype={"id":"string"}).rename(columns={"id":"station_id","dpcapacity":"capacity"})
    stations["capacity"]=pd.to_numeric(stations.capacity,errors="coerce")
    assert stations.station_id.is_unique
    out=S.root/"data/processed";out.mkdir(parents=True,exist_ok=True)
    trips.to_parquet(out/"trips.parquet",index=False);stations.to_parquet(out/"stations.parquet",index=False)
    manifest={"sources":records,"data_version":canonical_version(records),"input_rows":input_rows,"usable_trips":len(trips),"duplicates_removed":duplicate,"conflicting_ids":conflict,"invalid_rows":invalid,"stations":len(stations),"positive_capacity_stations":int((stations.capacity>0).sum()),"capacity_snapshot":"2026-09 retrospective scenario; not historical 2025 inventory","timezone":"America/Mexico_City (naive local source timestamps)"}
    dest=S.root/"references/data_manifest.json"
    if verify and dest.exists():
        old=json.loads(dest.read_text());assert old["data_version"]==manifest["data_version"],"Data version mismatch"
    dest.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2));return manifest
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true");ingest(parser.parse_args().verify)
