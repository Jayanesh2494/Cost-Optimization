import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

np.random.seed(42)

# Settings
end_date = datetime(2025, 9, 29)
dates = pd.date_range(end=end_date, periods=180)

providers = ['aws', 'azure']
regions = {
    'aws': ['us-east-1','us-west-2','eu-west-1','ap-south-1'],
    'azure': ['eastus','westeurope','southeastasia','centralindia']
}
resource_types = ['vm','container','db','storage']
services_map = {
    'vm': {'aws':'EC2','azure':'VirtualMachine'},
    'container': {'aws':'ECS','azure':'AKS'},
    'db': {'aws':'RDS','azure':'SQLDatabase'},
    'storage': {'aws':'S3','azure':'BlobStorage'}
}

resources = []
for provider in providers:
    for i in range(25):  # 25 resources per provider
        rtype = np.random.choice(resource_types, p=[0.5,0.2,0.15,0.15])
        resources.append({
            'provider': provider,
            'resource_id': f"{provider}-{i+1:03d}",
            'resource_type': rtype,
            'service': services_map[rtype][provider],
            'region': np.random.choice(regions[provider]),
            'vcpu': np.random.choice([1,2,4,8]),
            'memory_gb': np.random.choice([2,4,8,16,32]),
            'baseline_storage_gb': np.random.choice([10,50,100,250,500]),
            'is_spot': np.random.rand() < 0.15
        })

rows = []
for r in resources:
    for d_idx, dt in enumerate(dates):
        dow = dt.weekday()
        weekday_factor = 1.2 if dow < 5 else 0.7
        seasonal = 1 + 0.1 * math.sin(2*math.pi*(d_idx/30.0))

        cpu_util_pct = np.clip(
            (10 + r['vcpu']*5) * weekday_factor * seasonal * (1 + np.random.normal(0,0.1)),
            0, 100
        )
        cpu_hours = round((cpu_util_pct/100.0) * 24, 3)
        mem_hours = round(r['memory_gb'] * (cpu_hours/24.0) * np.random.uniform(0.5,1.1),3)
        storage_gb = round(r['baseline_storage_gb'] + d_idx*0.3 + np.random.normal(0,3),3)
        network_gb = round(np.random.normal(10,3) + cpu_hours*0.1, 3)

        cpu_cost = cpu_hours * r['vcpu'] * (0.04 if r['provider']=='aws' else 0.045)
        storage_cost = storage_gb * (0.025 if r['provider']=='aws' else 0.028)/30
        network_cost = max(network_gb,0) * 0.01
        overhead = np.random.uniform(0.1,0.5)
        spot_discount = 0.6 if r['is_spot'] else 1.0

        cost_usd = round((cpu_cost + storage_cost + network_cost + overhead)*spot_discount,4)

        rows.append({
            'date': dt.date().isoformat(),
            'provider': r['provider'],
            'resource_id': r['resource_id'],
            'resource_type': r['resource_type'],
            'service': r['service'],
            'region': r['region'],
            'vcpu': r['vcpu'],
            'memory_gb': r['memory_gb'],
            'cpu_util_percent': round(cpu_util_pct,3),
            'cpu_hours': cpu_hours,
            'memory_gb_hours': mem_hours,
            'storage_gb': storage_gb,
            'network_gb': max(network_gb,0),
            'is_spot': r['is_spot'],
            'idle_flag': cpu_util_pct < 5 or cpu_hours < 0.5,
            'cost_usd': cost_usd
        })

df = pd.DataFrame(rows)
df.to_csv("cloud_usage_full_synthetic.csv", index=False)

print("✅ Saved dataset as cloud_usage_full_synthetic.csv")
print(df.head())
