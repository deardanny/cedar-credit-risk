"""Metrics that preserve undefined quantities for zero-event cohorts."""
import numpy as np
from sklearn.metrics import mean_tweedie_deviance,mean_absolute_error,mean_squared_error
def money_metrics(y,p):
 y=np.asarray(y,dtype=float);p=np.clip(np.asarray(p,dtype=float),1e-8,None);total=float(y.sum());top=np.argsort(-p,kind='stable')[:int(np.ceil(len(p)/10))]
 return {'n':len(y),'actual_total':total,'predicted_total':float(p.sum()),'predicted_actual_ratio':float(p.sum()/total) if total else None,'actual_mean':float(y.mean()),'predicted_mean':float(p.mean()),'mae_dollars':float(mean_absolute_error(y,p)),'rmse_dollars':float(np.sqrt(mean_squared_error(y,p))),'tweedie_deviance':float(mean_tweedie_deviance(y/1000,p/1000,power=1.5)),'top_decile_amount_capture':float(y[top].sum()/total) if total else None}
