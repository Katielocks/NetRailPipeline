import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv
load_dotenv()

import rail_data


start_date = "2023-01-01"
end_date =  "2023-01-03"
rail_data.features.build_raw_weather_feature_frame(start_date=start_date,end_date=end_date)
rail_data.features.build_weather_features(start_date=start_date,end_date=end_date)

feature_file = Path(
    r"C:\Users\ktwhi\OneDrive - University of Bristol\Python Scripts\NetworkRailDelayModel\data\interim\weather\ELR_MIL=ABD_16\year=2023\month=1\day=1\features_0.parquet"
)
print(rail_data.io.read_cache(feature_file))