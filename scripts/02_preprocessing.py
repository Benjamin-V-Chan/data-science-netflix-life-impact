# 1. import pandas, pathlib, re
# 2. load raw DataFrame from outputs/raw_data.pkl
# 3. rename columns to snake_case
# 4. drop or fill missing values
# 5. convert types: release_year → int, average_rating → float, number_of_reviews → int
# 6. parse suggested_pct and suggested_flag from 'suggested_to_friends_family'
# 7. split minute_of_insight into insight_timestamp and insight_event
# 8. save cleaned DataFrame to outputs/cleaned_data.csv
