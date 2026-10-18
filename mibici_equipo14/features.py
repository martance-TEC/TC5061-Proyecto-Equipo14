"""Funciones puras de agregación y features sin usar eventos futuros."""
import numpy as np
import pandas as pd
from .config import SETTINGS as S
def hourly_counts(trips,stations):
    """Construye la malla estación-hora y cuenta salidas e ingresos por separado.

    Usa estaciones presentes como origen y destino y en el catálogo. Completa
    horas sin eventos con cero; la llegada se cuenta al finalizar el viaje.
    Es la base de P1 y P2 y de los perfiles de P3. La malla representa cobertura
    retrospectiva del año, no un calendario histórico de apertura de estaciones."""
    ids=sorted(set(trips.origin_id)&set(trips.destination_id)&set(stations.station_id))
    index=pd.MultiIndex.from_product([ids,pd.date_range(S.start,S.end,freq="h",inclusive="left")],names=["station_id","time"])
    counts=[]
    for station_col,time_col,name in [("origin_id","started_at","departures"),("destination_id","ended_at","arrivals")]:
        tmp=trips[[station_col,time_col]].copy();tmp["time"]=tmp[time_col].dt.floor("h")
        count=tmp.groupby([station_col,"time"]).size();count.index.names=["station_id","time"]
        counts.append(count.reindex(index,fill_value=0).rename(name))
    return pd.concat(counts,axis=1).reset_index()
def temporal_features(hourly):
    """Añade historia previa por estación y calendario conocido al predecir.

    Los rezagos de 1, 24 y 168 horas y medias de 24 y 168 horas se calculan
    después de desplazar una hora. Nunca incluyen la hora objetivo. Las filas
    iniciales incompletas se conservan con NaN para excluirlas al entrenar P1/P2.
    Requiere una malla horaria regular como la producida por hourly_counts."""
    x=hourly.sort_values(["station_id","time"]).copy()
    for col in ["departures","arrivals"]:
        g=x.groupby("station_id")[col]
        for lag in [1,24,168]:x[f"{col}_lag{lag}"]=g.shift(lag)
        for window in [24,168]:x[f"{col}_mean{window}"]=g.transform(lambda z:z.shift(1).rolling(window,min_periods=window).mean())
    x["hour"]=x.time.dt.hour;x["weekday"]=x.time.dt.dayofweek;x["month"]=x.time.dt.month
    return x
def pressure_label(deltas,capacity,initial_fraction=0.5):
    """Prefix balance for minute events in [cut,cut+3h); no inventory clipping."""
    if not np.isfinite(capacity) or capacity<=0:raise ValueError("Positive observed station capacity required")
    balances=np.r_[initial_fraction*capacity,initial_fraction*capacity+np.cumsum(deltas)]
    return int(balances.min()<=S.low_fraction*capacity or balances.max()>=S.high_fraction*capacity)
def capacity_labels(trips,hourly,stations,initial_fraction=0.5,return_direction=False):
    """Construye el objetivo proxy de P2 con capacidad observada del catálogo.

    Para cada hora reinicia un inventario hipotético initial_fraction * C y
    evalúa extremos de balance minuto a minuto durante las siguientes 3 horas.
    No recorta balances: una presión extrema intermedia también produce riesgo.
    Capacidad ausente/no positiva produce NaN y se excluye al entrenar. Si se
    solicita, devuelve la dirección bicycles, docks, both o none. Requiere la
    malla completa agrupada por estación y ordenada en el tiempo de hourly_counts.
    Este objetivo usa eventos futuros solo como etiqueta, nunca como predictor.
    No equivale a inventario real, demanda no atendida ni rebalanceo observado;
    el catálogo de 2026 se aplica retrospectivamente a los viajes de 2025."""
    events=pd.concat([trips[["origin_id","started_at"]].rename(columns={"origin_id":"station_id","started_at":"time"}).assign(delta=-1),trips[["destination_id","ended_at"]].rename(columns={"destination_id":"station_id","ended_at":"time"}).assign(delta=1)])
    events["time"]=events.time.dt.floor("min")
    events=events.groupby(["station_id","time"]).delta.sum()
    capacities=stations.set_index("station_id").capacity.to_dict();result=[];directions=[]
    for sid,group in hourly.groupby("station_id",sort=False):
        c=capacities.get(sid,np.nan)
        if not np.isfinite(c) or c<=0:
            result.extend([np.nan]*len(group));directions.extend(["capacity_missing"]*len(group));continue
        grid=pd.date_range(S.start,S.end,freq="min",inclusive="left")
        try:d=events.loc[sid].reindex(grid,fill_value=0).to_numpy(dtype=np.int16)
        except KeyError:d=np.zeros(len(grid),dtype=np.int16)
        # Each window starts with the same explicit hypothetical inventory.
        prefix=np.r_[0,np.cumsum(np.pad(d,(0,179)),dtype=np.int64)]
        ser=pd.Series(prefix)
        starts=np.arange(0,len(d),60)
        minimum=ser.rolling(181).min().to_numpy()[starts+180]-prefix[starts]
        maximum=ser.rolling(181).max().to_numpy()[starts+180]-prefix[starts]
        b=initial_fraction*c
        empty=minimum+b<=S.low_fraction*c;full=maximum+b>=S.high_fraction*c
        labels=(empty|full).astype(np.int8)
        directions.extend(np.select([empty&full,empty,full],["both","bicycles","docks"],default="none")[:len(group)])
        result.extend(labels[:len(group)])
    return (np.array(result),np.array(directions)) if return_direction else np.array(result)
def split_mask(times,split,horizon=1):
    """Selecciona filas cuyo horizonte completo queda dentro del periodo.

    La hora inicial es inclusiva y el final del objetivo puede coincidir con
    el límite exclusivo del periodo. Purga objetivos que lo sobrepasarían:
    1 hora para P1 y 3 para P2. Las fechas proceden de Settings centralizado."""
    bounds={"train":(S.start,S.train_end),"val":(S.train_end,S.val_end),"test":(S.val_end,S.end)}
    lo,hi=map(pd.Timestamp,bounds[split]);return (times>=lo)&(times+pd.Timedelta(hours=horizon)<=hi)
def station_profiles(hourly,start,end,ids=None):
    """Resume patrones históricos de uso para P3 dentro de [start, end).

    Genera 96 proporciones de actividad (salidas/ingresos por 24 horas y
    laborable/fin de semana), volumen diario logarítmico, fracción de salidas
    de fin de semana y balance neto: 99 variables. ids limita estaciones.
    No utiliza coordenadas; se añaden después para mostrar los grupos en mapa.
    Calcular cada periodo por separado evita mezclar validación/prueba en train.
    El escalado y KMeans se ajustan fuera de esta función, solo con train."""
    h=hourly[(hourly.time>=start)&(hourly.time<end)].copy()
    if ids is not None:h=h[h.station_id.isin(ids)]
    h["hour"]=h.time.dt.hour;h["weekend"]=(h.time.dt.dayofweek>=5).astype(int)
    out=[]
    for sid,g in h.groupby("station_id"):
        total=g.departures.sum()+g.arrivals.sum();row={"station_id":sid}
        for col in ["departures","arrivals"]:
            profile=g.groupby(["weekend","hour"])[col].sum().reindex(pd.MultiIndex.from_product([[0,1],range(24)]),fill_value=0)
            row.update({f"{col}_{i}":float(v/max(total,1)) for i,v in enumerate(profile)})
        row.update(log_volume=float(np.log1p(total/len(g)*24)),weekend_share=float(g.loc[g.weekend==1,"departures"].sum()/max(g.departures.sum(),1)),balance=float((g.arrivals.sum()-g.departures.sum())/max(total,1)))
        out.append(row)
    return pd.DataFrame(out).set_index("station_id").sort_index()
