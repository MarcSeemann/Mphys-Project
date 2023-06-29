
import pandas as pd



data_frame_0 = pd.read_hdf('test_final_fn_0.h5')
    
data_frame_1 = pd.read_hdf('test_final_fn_1.h5')
data_frame_1 = data_frame_1.iloc[: , 1:]
data_frame_1 = data_frame_1.add_suffix('_1')

data_frame_2 = pd.read_hdf('test_final_fn_2.h5')
data_frame_2 = data_frame_2.iloc[: , 1:]
data_frame_2 = data_frame_2.add_suffix('_2')

data_frame_3 = pd.read_hdf('test_final_fn_3.h5')
data_frame_3 = data_frame_3.iloc[: , 1:]
data_frame_3 = data_frame_3.add_suffix('_3')

data_frame_4 = pd.read_hdf('test_final_fn_4.h5')
data_frame_4 = data_frame_4.iloc[: , 1:]
data_frame_4 = data_frame_4.add_suffix('_4')

data_frame_5 = pd.read_hdf('test_final_fn_5.h5')
data_frame_5 = data_frame_5.iloc[: , 1:]
data_frame_5 = data_frame_5.add_suffix('_5')

data_frame_6 = pd.read_hdf('test_final_fn_6.h5')
data_frame_6 = data_frame_6.iloc[: , 1:]
data_frame_6 = data_frame_6.add_suffix('_6')

data_frame_7 = pd.read_hdf('test_final_fn_7.h5')
data_frame_7 = data_frame_7.iloc[: , 1:]
data_frame_7 = data_frame_7.add_suffix('_7')

data_frame_8 = pd.read_hdf('test_final_fn_8.h5')
data_frame_8 = data_frame_8.iloc[: , 1:]
data_frame_8 = data_frame_8.add_suffix('_8')

data_frame_9 = pd.read_hdf('test_final_fn_9.h5')
data_frame_9 = data_frame_9.iloc[: , 1:]
data_frame_9 = data_frame_9.add_suffix('_9')

data_frame_10 = pd.read_hdf('test_final_fn_10.h5')
data_frame_10 = data_frame_10.iloc[: , 1:]
data_frame_10 = data_frame_10.add_suffix('_10')

data_frame_11 = pd.read_hdf('test_final_fn_11.h5')
data_frame_11 = data_frame_11.iloc[: , 1:]
data_frame_11 = data_frame_11.add_suffix('_11')

data_frame_12 = pd.read_hdf('test_final_fn_12.h5')
data_frame_12 = data_frame_12.iloc[: , 1:]
data_frame_12 = data_frame_12.add_suffix('_12')

data_frame_13 = pd.read_hdf('test_final_fn_13.h5')
data_frame_13 = data_frame_13.iloc[: , 1:]
data_frame_13 = data_frame_13.add_suffix('_13')

data_frame_14 = pd.read_hdf('test_final_fn_14.h5')
data_frame_14 = data_frame_14.iloc[: , 1:]
data_frame_14 = data_frame_14.add_suffix('_14')

data_frame_15 = pd.read_hdf('test_final_fn_15.h5')
data_frame_15 = data_frame_15.iloc[: , 1:]
data_frame_15 = data_frame_15.add_suffix('_15')

data_frame_16 = pd.read_hdf('test_final_fn_16.h5')
data_frame_16 = data_frame_16.iloc[: , 1:]
data_frame_16 = data_frame_16.add_suffix('_16')

data_frame_17 = pd.read_hdf('test_final_fn_17.h5')
data_frame_17 = data_frame_17.iloc[: , 1:]
data_frame_17 = data_frame_17.add_suffix('_17')

data_frame_18 = pd.read_hdf('test_final_fn_18.h5')
data_frame_18 = data_frame_18.iloc[: , 1:]
data_frame_18 = data_frame_18.add_suffix('_18')

data_frame_19 = pd.read_hdf('test_final_fn_19.h5')
data_frame_19 = data_frame_19.iloc[: , 1:]
data_frame_19 = data_frame_19.add_suffix('_19')

data_frame_20 = pd.read_hdf('test_final_fn_20.h5')
data_frame_20 = data_frame_20.iloc[: , 1:]
data_frame_20 = data_frame_20.add_suffix('_20')

data_frame_21 = pd.read_hdf('test_final_fn_21.h5')
data_frame_21 = data_frame_21.iloc[: , 1:]
data_frame_21 = data_frame_21.add_suffix('_21')

data_frame_22 = pd.read_hdf('test_final_fn_22.h5')
data_frame_22 = data_frame_22.iloc[: , 1:]
data_frame_22 = data_frame_22.add_suffix('_22')

results = pd.concat([data_frame_0,data_frame_1, data_frame_2, data_frame_3, data_frame_4, data_frame_5, data_frame_6, data_frame_7, data_frame_8, data_frame_9, data_frame_10, data_frame_11, data_frame_12, data_frame_13, data_frame_14, data_frame_15, data_frame_16, data_frame_17, data_frame_18, data_frame_19, data_frame_20, data_frame_21, data_frame_22],axis=1)
# results = pd.concat([data_frame_0,data_frame_1, data_frame_2, data_frame_3, data_frame_4, data_frame_5, data_frame_6, data_frame_7, data_frame_8, data_frame_9, data_frame_10, data_frame_11, data_frame_12, data_frame_13],axis=1)

results.to_hdf('combined_calib_test.h5', key='test', index=False)
print('Finished')

