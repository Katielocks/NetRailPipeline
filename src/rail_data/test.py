import rail_io
import datetime as dt

start_date = dt.datetime(2023,1,1)
end_date =  dt.datetime(2024,1,2)
rail_io.get_datasets(start_date=start_date,end_date=end_date)