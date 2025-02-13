import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np


def buckets(file_path, target = "Close_denoised_standardized", horizon = 12, threshold_column=None):
    df = pd.read_csv(file_path, parse_dates = True, index_col = 0)
    df["DistanceToEMA60"] = df["DistanceToEMM60"]
    df.dropna(inplace=True)

    def atr(target, horizon = 12, threshold_column = None, q1 = 0, q2 = 0.1):
        i = 384
        TP = [0 for k in range(horizon)]
        TN = [0 for k in range(horizon)]
        FP = [0 for k in range(horizon)]
        FN = [0 for k in range(horizon)]
        if threshold_column is not None:
            low_threshold = df[threshold_column].quantile(q1)
            high_threshold = df[threshold_column].quantile(q2)

        # -13 is used to avoid predicting the next 12 hours for our last data point.
        # Our last data point represents an hour. Starting the prediction at this hour is not possible because we could not verify if the prediction is correct or not. 
        while i < len(df)-13:
            try:
                index = df.index[i]
                # Filter out all the rows that have a value in their threshold_column that is above the high_threshold or below the low_threshold.
                if threshold_column is not None and (df[threshold_column].iloc[i] > high_threshold or df[threshold_column].iloc[i] < low_threshold):
                    i += 1
                    continue
                # If the value in the threshold column is withing the range of the high and low thresholds, we proceed.
                base = df[target].iloc[i] # we extract the value of the close denoised standardized at the current index.
                signs = list()
                for j in range(1, horizon+1):
                    y = df[target].iloc[i+j]
                    signs.append(int(np.sign(base-y)))
                # we jump to the next hour
                i+=1
                pred_signs = ast.literal_eval(df["SIGN"].iloc[i])
                for j in range(horizon):
                    if pred_signs[j] == 1 and signs[j] == -1:
                        TN[j] += 1
                    elif pred_signs[j] == 1:
                        TP[j] += 1
                    elif pred_signs[j] == -1 and signs[j] == -1:
                        FP[j] += 1
                    elif pred_signs[j] == -1:
                        FN[j] += 1
            except Exception as e:
                print(i)
                break
        return list(map(lambda x: x[0]/(x[0]+x[1]), zip(TP, FP)))
    
    # compute the precision for each ventile
    first_df = []
    for ventile in range(20):
        temp = pd.DataFrame(atr(target, horizon, threshold_column, q1 = ventile/20, q2 = (ventile+1)/20)).T
        temp.to_csv(f"first_df_temp_{ventile}.csv", index="False")
        first_df.append(temp)
    
    # This dataframe is for the baseline
    second_df = pd.DataFrame(atr(target, horizon)).T
    second_df.to_csv("second_df_test.csv", index="False")
    final_df = pd.concat(first_df + [second_df], axis=0)
    final_df.to_csv("final_df_test.csv")

    return final_df


import matplotlib.pyplot as plt

# Call the buckets function with horizon=12 using ATR_10 as threshold_column.
df = buckets("/Users/baudoincoispeau/Documents/Keensight/TF4FIP/analysis/es_future_final_moirai.csv", horizon=12, threshold_column="ATR_10")

# The buckets function returns 21 rows: 20 ventiles (quantile buckets) + 1 baseline.
# We drop the baseline (the last row).
df = df.iloc[:-1]

# Transpose so that rows represent horizon steps (H=1 to H=12) and columns represent the ventiles.
df = df.T

# Reset the index to represent horizon steps 1 to 12.
df.index = pd.RangeIndex(start=1, stop=df.index.stop + 1, step=1)

# Set column names for the 20 ventiles (20-quantiles).
bucket_labels = [f"ATR_10 {i*5}-{(i+1)*5}" for i in range(20)]
df.columns = bucket_labels


# Extract the precision values for horizons H=1, H=3, and H=12.
prec_H1 = df.loc[1]   # H=1 (first forecast step)
prec_H3 = df.loc[3]   # H=3 (third forecast step)
prec_H12 = df.loc[12] # H=12 (twelfth forecast step)

# Plot the results.
plt.figure(figsize=(20, 10))
plt.plot(bucket_labels, prec_H1, linestyle='-', marker=None, label="H=1")    # plain line for H=1
plt.plot(bucket_labels, prec_H3, linestyle='--', marker=None, label="H=3")   # dash line for H=3
plt.plot(bucket_labels, prec_H12, linestyle='-', marker='o', label="H=12")   # point line for H=12

plt.title("ES Future - Precision per buckets - Moirai")
plt.xlabel("Ventile for ATR_10")
plt.ylabel("Average precision per ventile")
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
