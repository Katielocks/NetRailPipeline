import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv
load_dotenv()

import rail_data

start_date = "2023-01-01"
end_date =  "2023-01-03"
rail_data.features.build_raw_weather_feature_frame(start_date,end_date)

