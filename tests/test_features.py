import numpy as np
import pandas as pd
import pytest
from mibici_equipo14.features import pressure_label,temporal_features,split_mask,hourly_counts
from mibici_equipo14.dataset import canonical_version
def test_capacity_changes_risk():
    assert pressure_label([-8],20)==1
    assert pressure_label([-8],40)==0
def test_prefix_extreme_not_only_final():
    assert pressure_label([-9,9],20)==1
def test_requires_observed_positive_capacity():
    with pytest.raises(ValueError):pressure_label([-1],0)
def test_future_event_cannot_change_closed_lag():
    x=pd.DataFrame({'station_id':['2']*200,'time':pd.date_range('2025-01-01',periods=200,freq='h'),'departures':np.arange(200),'arrivals':0})
    before=temporal_features(x);x.loc[180:,'departures']=99999;after=temporal_features(x)
    assert before.loc[180,'departures_lag1']==after.loc[180,'departures_lag1']==179
    assert before.loc[180,'departures_mean168']==after.loc[180,'departures_mean168']
def test_horizon_does_not_cross_training_boundary():
    t=pd.Series(pd.to_datetime(['2025-08-31 20:00','2025-08-31 22:00','2025-09-01 00:00']))
    assert split_mask(t,'train',3).tolist()==[True,False,False]
def test_arrival_counted_at_end_time():
    trips=pd.DataFrame({'origin_id':['2'],'destination_id':['2'],'started_at':pd.to_datetime(['2025-01-01 00:50']),'ended_at':pd.to_datetime(['2025-01-01 01:10'])})
    h=hourly_counts(trips,pd.DataFrame({'station_id':['2']}));assert h.iloc[0].departures==1;assert h.iloc[0].arrivals==0;assert h.iloc[1].arrivals==1
def test_hash_is_stable_for_key_order():
    assert canonical_version([{'b':2,'a':1}])==canonical_version([{'a':1,'b':2}])
