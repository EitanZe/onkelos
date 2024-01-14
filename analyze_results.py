import pandas as pd

results = pd.read_csv("Verses with Sig Words NB weights no eng.csv")
init_df = {"Perek":["1:01"], "# verses":[0], "Total jer attr":[0],
                        ">90% prob.":[0], ">85% prob.":[0], ">75% prob.":[0]}
quant_df = pd.DataFrame(init_df)

for index, row in results.iterrows():
    p = row["Verse"][0:4]
    perek = quant_df.loc[quant_df.index[-1],"Perek"]
    
    # check if looking at a new perek, if so, add new row to df
    if p != perek:
        new_row = {"Perek":p, "# verses":0, "Total jer attr":0,
                        ">90% prob.":0, ">85% prob.":0, ">75% prob.":0}
        quant_df.loc[len(quant_df)] = new_row

    quant_df.loc[quant_df.index[-1],"# verses"] += 1 # increase total # of verses counter

    if float(row["Yerushalmi %"]) > 0.5: # if above 0.5, verse is attributed to yerushalmi/jer
        quant_df.loc[quant_df.index[-1],"Total jer attr"] += 1
    if float(row["Yerushalmi %"]) > 0.75:
        quant_df.loc[quant_df.index[-1],">75% prob."] += 1
    if float(row["Yerushalmi %"]) > 0.85:
        quant_df.loc[quant_df.index[-1],">85% prob."] += 1
    if float(row["Yerushalmi %"]) > 0.9:
        quant_df.loc[quant_df.index[-1],">90% prob."] += 1

# print(quant_df.head())
quant_df.to_csv("Number of Yerushalmi Verses Per Perek.csv", index=False)
    