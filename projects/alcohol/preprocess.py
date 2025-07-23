import pandas

# From the paper: Dora et al (2025) https://osf.io/preprints/psyarxiv/c7q9p_v1
# Preprocess from original data at https://osf.io/j9bkq/files/osfstorage

# psychomotor vigilance task to measure response time
df1 = pandas.read_csv("pvtData1.csv")
df1['after_manipulation'] = False
df2 = pandas.read_csv("pvtData2.csv")
df2['after_manipulation'] = True
df = pandas.concat([df1, df2]).sort_values("Subject", kind="stable")
df.to_csv("pvt.csv", index=False)

# All other data
df = pandas.read_csv("data.csv")

# Participant demographics, self-report preferences for alcoholic and non-alcoholic
# drinks, willingness to pay data, stress ratings, mood ratings, and alcohol use
# disorders identification test (AUDIT)
df.drop(["trial_number", "LeftImage", "RightImage", "Choice", "Choice2", "RT", "pre_post"], axis=1).drop_duplicates().to_csv("participants.csv", index=False)

# Simplified version of the data file
df[['pid', 'trial_number', 'Choice2', 'RT', 'pre_post', 'Condition', 'LeftImage', 'RightImage', 'Choice']].rename({"Choice2": "choice", "Choice": "choice_identity"}, axis=1).to_csv("trials.csv", index=False)
