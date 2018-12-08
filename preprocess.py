import numpy as np
import pandas as pd
import math
from scipy.stats import mode
from sklearn import preprocessing as pre
from copy import deepcopy
from tqdm import tqdm

DF = pd.read_pickle('/data/ards_temp/ards/code/savedData/eligDF.pkl').sort_values(['id_x'])
DF = DF.reset_index(drop=True)


def id_to_idx(df3,ids):
    id_dict = {}
    if (len(ids) == 0):
        i = 0
        for id in sorted(df3['id'].unique()):
            id_dict[id] = i
            i += 1
    else:
        i = 0
        for id in sorted(ids):
            id_dict[id] = i
            i += 1
            
    return id_dict


def variables(all_columns,cat_or_cont):
    cols = all_columns
    cat_cols = [c for c in all_columns if c.lower()[-4:] == "flag"]
    cat_cols.append('location')
    if cat_or_cont == "cat":
        cols = cat_cols
    elif cat_or_cont == "cont":
        cols = [c for c in all_columns if c not in cat_cols]
        cols.remove('id')
        cols.remove('time')
        li = ['alert','unresponsive','sedated','oriented','invasive','noninvasive','hfnc','dialysis','eligible','encounterid']
        for x in li:
            cols.remove(x)
        cat_cols.remove('location')
        for c in cat_cols:
            cols.remove(c[:-5])
    return cols


def format_long_data_cat(df3,cols_to_feat,time):
    #First 6 hours
    new_df3 = df3.copy()
    new_df3 = create_newdf3(df3,DF,time)
    id_list = sorted(list(set(list(new_df3['id']))))

    cols = variables(new_df3.columns,"cat")
    
    ###CREATE DICTIONARY###
    suffix = ['_ll', '_l', '_n', '_h', '_hh', '_missing']
    location_suffix = ['_ED', '_ICU', '_floor', '_Proceduce', '_8D', '_Missing']
    
    i = 0
    for col in cols:
        for j in range(i, i + 6):
            if col == 'location':
                cols_to_feat[j] = col + location_suffix[j - i]
            else:
                cols_to_feat[j] = col + suffix[j - i]
        i += 6

    ###CREATE DICTIONARY###
        
    cat_arr = np.zeros((len(id_list),len(cols),6))
    dic = {'LL':[1,0,0,0,0,0],'L':[0,1,0,0,0,0],'N':[0,0,1,0,0,0],'H':[0,0,0,1,0,0],'HH':[0,0,0,0,1,0],'A':[0,0,0,0,0,1]}
    loc_dic = {'ED':[1,0,0,0,0,0],'ICU':[0,1,0,0,0,0],'floor':[0,0,1,0,0,0],'Procedure':[0,0,0,1,0,0], '8D': [0,0,0,0,1,0]}
    i = 0
    for id in id_list:
        temp = new_df3[new_df3['id']==id]
        patient_feature_vector = np.zeros((len(cols),6))
        j = 0
        for c in cols:
            column_vector = np.zeros(6)
            for idx,row in temp.iterrows():
                if (c == 'location'):
                    #Location is known
                    if (not type(row[c]) is float):
                        column_vector = np.logical_or(column_vector,loc_dic[row[c]]).astype(int)
                    #Location is unknown
                    else:
                        column_vector[5] = 1
                #If continuous entry exists
                elif not math.isnan(row[c[:-5]]):
                    #Normal level: N
                    if (type(row[c]) is float):
                        column_vector = np.logical_or(column_vector,dic['N']).astype(int)
                    #Abnormal level: LL,L,H,HH
                    else:
                        column_vector = np.logical_or(column_vector,dic[row[c]]).astype(int)
                #If data is missing
                else:
                    column_vector[5] =  1

            if (sum(column_vector[:5]) and column_vector[5]):
                column_vector[5] = 0

            patient_feature_vector[j] = column_vector.astype(int)
            j += 1

        cat_arr[i] = patient_feature_vector
        i += 1
    
    #Make it 2D
    cat_arr = np.reshape(cat_arr,(cat_arr.shape[0],-1))
    
    li = variables(new_df3.columns,'cat')
    li.remove('location')
    final = np.zeros((len(id_list), len(li)))
    i = 0
    for id in id_list:
        temp = new_df3[new_df3['id'] == id]
        j = 0
        ma = -1
        for var in li:
            temp1 = temp[var[:-5]]
            final[i][j] = temp1.count()
            j += 1
        i += 1

    last = np.zeros((final.shape[0],final.shape[1],5))
    for i in range(final.shape[1]):
        mat = pd.qcut(pd.Series(final[:,i]),5,duplicates='drop',labels=False).as_matrix()
        for j in range(mat.shape[0]):
            last[j,i,mat[j]] = 1

    last = last.reshape((final.shape[0],-1))
    ind = len(cols_to_feat)
    dic = {}
    count = 0
    for i in range(final.shape[1]):
        total = np.max(pd.qcut(pd.Series(final[:,i]),5,duplicates='drop',labels=False).as_matrix()) + 1
        count += total
        bins = list(pd.qcut(pd.Series(final[:,i]),5,duplicates='drop',retbins=True)[1])
        for j in range(total):
            dic[ind] = li[i][:-5] + '_count' + '_(' + str(bins[j]) + ',' + str(bins[j+1]) + ']' + '_b' + str(j+1) + '_t' + str(total)
            ind += 1
    
    #Fix array where column sums to 0 caused by less than 5 quintile bins
    li = []
    for i in range(last.shape[1]):
        if (np.sum(last,axis=0)[i] == 0):
            li.append(i)

    #Delete dictionary elements 
    li = sorted(li,reverse=True)
    for idx in li:
        last = np.delete(last,np.s_[idx],axis=1)
    
    cols_to_feat = {**cols_to_feat,**dic}
    cat_arr = np.c_[cat_arr,last]
    
    print('FINISHED CATEGORICAL VARIABLES!') 

    return cat_arr,cols_to_feat


def format_long_data_cont(df3,cols_to_feat,time):
    new_df3 = df3.copy()
    new_df3 = create_newdf3(df3,DF,time)
    id_list = sorted(list(set(list(new_df3['id']))))
    cols = variables(new_df3.columns,"cont")
    
    ###CREATE DICTIONARY###
    suffix1 = ['_min', '_max', '_mean', '_median', '_std', '_iqr', '_missing']
    suffix2 = ['_b1','_b2','_b3','_b4','_b5']
    
    temp_dict = {}
    i = len(cols_to_feat)
    for col in cols:
        for j in range(6):
            for k in range(5):
                temp_dict[i] = str(col)+suffix1[j]+suffix2[k]
                i += 1
        temp_dict[i] = str(col)+suffix1[-1]
        i += 1
    ###CREATE DICTIONARY###
    
    #Create 6 rows (one for each statistic) for each patient
    new = pd.DataFrame(columns=['id']+cols)
    for id in id_list:
        for i in range(6):
            new.loc[len(new)] = np.append([id],np.zeros(len(cols)))
    new['id'] = new['id'].astype(int)
    new = new.sort_values(['id'])
    #Min,max,mean,median,stdev.,IQR
    min_idx = 0
    max_idx = 1
    mean_idx = 2
    median_idx = 3
    std_idx = 4
    iqr_idx = 5
    
    print("THIS IS SLOW! PATIENCE IS A VIRTUE!")
    
    for id in id_list:
        min_arr = np.zeros(len(cols))
        max_arr = np.zeros(len(cols))
        mean_arr = np.zeros(len(cols))
        median_arr = np.zeros(len(cols))
        std_arr = np.zeros(len(cols))
        iqr_arr = np.zeros(len(cols))
        j = 0
        temp = new_df3[new_df3['id'] == id]
        for c in cols:
            min_arr[j] = temp[c].min()
            max_arr[j] = temp[c].max()
            mean_arr[j] = temp[c].mean()
            median_arr[j] = temp[c].median()
            std_arr[j] = np.std(temp[c])
            iqr_arr[j] = temp[c].quantile(0.75) - temp[c].quantile(0.25)
            j += 1
        new.loc[min_idx,:] = np.append([id],min_arr)
        new.loc[max_idx,:] = np.append([id],max_arr)
        new.loc[mean_idx,:] = np.append([id],mean_arr)
        new.loc[median_idx,:] = np.append([id],median_arr)
        new.loc[std_idx,:] = np.append([id],std_arr)
        new.loc[iqr_idx,:] = np.append([id],iqr_arr)
        min_idx += 6
        max_idx += 6
        mean_idx += 6
        median_idx += 6
        std_idx += 6
        iqr_idx += 6
    
    new = new.sort_index()
    print('SLOW PART DONE!')
    
    NUM_SUMMARY_STATS = 6
    NUM_BINS = 5
    #2D final array
    cont_arr = np.zeros((len(id_list),len(cols) * (NUM_SUMMARY_STATS * NUM_BINS + 1)))
    k = 0
    for c in cols:
        col = new[c]
        min_bins = pd.qcut(col[0::6],5,labels=False,duplicates='drop').as_matrix()
        max_bins = pd.qcut(col[1::6],5,labels=False,duplicates='drop').as_matrix()
        mean_bins = pd.qcut(col[2::6],5,labels=False,duplicates='drop').as_matrix()
        median_bins = pd.qcut(col[3::6],5,labels=False,duplicates='drop').as_matrix()
        std_bins = pd.qcut(col[4::6],5,labels=False,duplicates='drop').as_matrix()
        iqr_bins = pd.qcut(col[5::6],5,labels=False,duplicates='drop').as_matrix()
        arr = np.zeros((len(id_list),NUM_SUMMARY_STATS * NUM_BINS + 1))
        for i in range(len(min_bins)):
            if (math.isnan(min_bins[i]) or math.isnan(max_bins[i]) or math.isnan(mean_bins[i]) or math.isnan(median_bins[i]) or \
                math.isnan(std_bins[i]) or math.isnan(iqr_bins[i])):
                arr[i,-1] = 1
            else:
                arr[i,int(min_bins[i])] = 1
                arr[i,int(max_bins[i])+5] = 1
                arr[i,int(mean_bins[i])+10] = 1
                arr[i,int(median_bins[i])+15] = 1
                arr[i,int(std_bins[i])+20] = 1
                arr[i,int(iqr_bins[i])+25] = 1
        cont_arr[:,k:k+31] = arr
        k += 31

    const = len(cols_to_feat)
    i = const
    for col in cols:
        if col == 'vasso' or col == 'dopa' or col == 'dobu' or col == 'mil' or col == 'plt_transf' \
            or col == 'rbc_transf' or col == 'epi' or col == 'ffp_transf':
            i += 31
            continue

        while (i < len(temp_dict)+const):
            val = temp_dict[i]
            if col == 'pf' and 'pf_calc' in val:
                break
            elif col in val:
                if 'min' in val:
                    print('1.',val)
                    bins = list(pd.qcut(new[col][0::6],5,duplicates='drop',retbins=True)[1])
                elif 'max' in val:
                    print('2.',val)
                    bins = list(pd.qcut(new[col][1::6],5,duplicates='drop',retbins=True)[1])
                elif 'mean' in val:
                    print('3.',val)
                    bins = list(pd.qcut(new[col][2::6],5,duplicates='drop',retbins=True)[1])
                elif 'median' in val:
                    print('4.',val)
                    bins = list(pd.qcut(new[col][3::6],5,duplicates='drop',retbins=True)[1])
                elif 'std' in val:
                    print('5.',val)
                    bins = list(pd.qcut(new[col][4::6],5,duplicates='drop',retbins=True)[1])
                elif 'iqr' in val:
                    print('6.',val)
                    bins = list(pd.qcut(new[col][5::6],5,duplicates='drop',retbins=True)[1])
                else:
                    temp_dict[i] = val
                    i += 1
                    continue
                for j in range(i,i+len(bins)-1):
                    temp_dict[j] = val[:-2]+'('+str(bins[j-i])+','+str(bins[j-i+1])+']'+'_b'+str(j-i+1)+'_t'+str(len(bins)-1)
                i += 5
            else:
                break
    #Fix array where column sums to 0 caused by less than 5 quintile bins
    i = 0
    li = []
    for i in range(cont_arr.shape[1]):
        if (np.sum(cont_arr,axis=0)[i] == 0):
            li.append(i+const)

    #Delete dictionary elements 
    li = sorted(li,reverse=True)
    for idx in li:
        cont_arr = np.delete(cont_arr,np.s_[idx-const],axis=1)
        del temp_dict[idx]

    #Fix the dictionary so that indices are aligned
    vals = []
    for k,v in temp_dict.items():
        vals.append(v)

    new_dic = {}
    i = const
    for v in vals:
        new_dic[i] = v
        i += 1
        
    total_cols = ['iv_in','urine_out']
    li1 = []
    li2 = []
    flag = False
    for c in total_cols:
        for id in id_list:
            if not flag:
                li1.append(np.sum(new_df3[new_df3['id'] == id][c]))
            else:
                li2.append(np.sum(new_df3[new_df3['id'] == id][c]))
        flag = True
    
    idx1 = pd.qcut(pd.Series(li1),5,duplicates='drop',labels=False).as_matrix()
    bins1 = list(pd.qcut(pd.Series(li1),5,duplicates='drop',retbins=True)[1])
    idx2 = pd.qcut(pd.Series(li2),5,duplicates='drop',labels=False).as_matrix()
    bins2 = list(pd.qcut(pd.Series(li2),5,duplicates='drop',retbins=True)[1])
    mat1 = np.zeros((len(idx1),len(bins1)-1))
    mat2 = np.zeros((len(idx2),len(bins2)-1))
    for i in range(len(idx1)):
        mat1[i,int(idx1[i])] = 1
        mat2[i,int(idx2[i])] = 1

    i = len(new_dic)+const
    for j in range(i,i+len(bins1)-1):
        new_dic[j] = total_cols[0]+'_total_'+'('+str(bins1[j-i])+','+str(bins1[j-i+1])+']'+'_b'+str(j-i+1)+'_t'+str(len(bins1)-1)
    i = len(new_dic)+const
    for j in range(i,i+len(bins2)-1):
        new_dic[j] = total_cols[1]+'_total_'+'('+str(bins2[j-i])+','+str(bins2[j-i+1])+']'+'_b'+str(j-i+1)+'_t'+str(len(bins2)-1)

    cont_arr = np.c_[cont_arr,mat1]
    cont_arr = np.c_[cont_arr,mat2]
    
    cols_to_feat = {**cols_to_feat,**new_dic}
    return cont_arr,cols_to_feat


def format_med_data(df3,df4,cols_to_feat,time):
    new_df4 = df4.copy()
    new_df4 = create_newdf4(df4,DF,time)
    new_df4 = new_df4.dropna(subset=['vaclasscode'])
    #Medicine exclusion list
    li = ['DE350','DE400','DE500','DE700','DX101','DX102','GA101','GA105','GA108','GA110',\
          'GA201','GA202','GA203','GA204','GA205','HA000','IR100','MS300',\
          'NT300','OP109','OP600','OP800','OR300','OR500','PH000','XX000']
    for med in li:
        new_df4 = new_df4[new_df4['vaclasscode'] != med]
    
    temp = create_newdf3(df3,DF,time)
    id_list = sorted(list(temp['id'].unique()))
    meds = sorted(list(new_df4['vaclasscode'].unique()))
    counts = np.zeros((len(id_list),len(meds)))
    medid_list = sorted(list(new_df4['id'].unique()))
    
    ###CREATE DICTIONARY###
    i = len(cols_to_feat)
    for col in meds:
        cols_to_feat[i] = col
        i += 1
    ###CREATE DICTIONARY###
    
    #Map each unique medicine to its own index so that a column in the matrix corresponds to that
    #medicine
    le = pre.LabelEncoder()
    le.fit(meds)
    
    id_dictionary = id_to_idx(temp,id_list)
    
    for id in medid_list:
        temp = new_df4[new_df4['id'] == id]
        hot = le.transform(temp['vaclasscode'])
        for idx in hot:
            if (counts[id_dictionary[id],idx] == 0):
                counts[id_dictionary[id],idx] = 1
            else:
                continue

    return counts,cols_to_feat


def format_output(df2,ids):
    if (len(ids) == 0):
        y = np.zeros(len(df2['id'].unique()))
        i = 0
        for id in sorted(df2['id'].unique()):
            temp = df2[df2['id'] == id]
            #Take a majority vote among physicians' diagnoses
            label = mode(temp['ards_pt'])
            y[i] = label[0][0]
            i += 1
    else:
        y = np.zeros(len(ids))
        i = 0
        for id in sorted(ids):
            temp = df2[df2['id'] == id]
            #Take a majority vote among physicians' diagnoses
            label = mode(temp['ards_pt'])
            y[i] = label[0][0]
            i += 1
    return y


def format_baseline_data(complete_data,df1,df3,cols_to_feat,time):
    #GENDER INFO
    temp1 = df1.copy()

    new_df3 = create_newdf3(df3,DF,time)
    temp1 = temp1[temp1['id'].isin(new_df3['id'].unique())]
    arr = temp1['gendercode'].as_matrix()
    arr[arr == 'M'] = 0
    arr [arr != 0] = 1
    complete_data = np.c_[complete_data, arr]
    print('2. Shape of final array: ',complete_data.shape)
    cols_to_feat[len(cols_to_feat)] = 'gendercode'
    
    #AGE INFO
    temp = df1.copy()
    temp = temp[temp['id'].isin(new_df3['id'].unique())]
    arr = temp['ageinyears'].as_matrix()
    indices = pd.qcut(arr,5,labels=False)
    mat = np.zeros((len(arr),5))
    j = 0
    for i in indices:
        mat[j,i] = 1
        j += 1
    complete_data = np.c_[complete_data, mat]
    print('3. Shape of final array: ',complete_data.shape)
    
    ###CREATE DICTIONARY###
    bins = list(pd.qcut(temp['ageinyears'],5,duplicates='drop',retbins=True)[1])
    i = len(cols_to_feat)
    for j in range(i,i+len(bins)-1):
        cols_to_feat[j] = 'ageinyears_'+'('+str(bins[j-i])+','+str(bins[j-i+1])+']'+'_b'+str(j-i+1)+'_t'+str(len(bins)-1)

    #RACE INFO
    print('Race: ')
    for race in df1['racename'].unique():
        temp3 = df1.copy()
        temp3 = temp3[temp3['id'].isin(new_df3['id'].unique())]
        temp3['racename'].replace('Patient Refused','Other',inplace=True)
        temp3['racename'].replace('Unknown','Other',inplace=True)
        temp3['racename'].fillna('Other',inplace=True)
        arr = temp3['racename'].as_matrix()
        arr[arr == race] = 1
        arr[arr != 1] = 0
        if (race != 'Patient Refused' and race != 'Unknown' and not type(race) is float):
            complete_data = np.c_[complete_data, arr]
        else:
            continue
    print('4. Shape of final array: ',complete_data.shape)
    
    ###CREATE DICTIONARY###
    k = len(cols_to_feat)
    
    races = list(df1['racename'].unique())
    for x in races:
        if type(x) is float:
            races.remove(x)
    races.remove('Patient Refused')
    races.remove('Unknown')
    races = ['_' + x for x in races]
    
    for i in range(k,k+6):
        cols_to_feat[i] = 'racename'+races[i-k]
    
    return complete_data,cols_to_feat


def getFeaturesPath():
    return 'features_test.npy'


def getLabelsPath():
    return 'labels_test.npy'


def format_design_matrix(df1,df2,df3,df4,time):
    cols_to_feat = {}
    print('Initializing matrix for categorical variables...')
    cat_arr,cols_to_feat = format_long_data_cat(df3,cols_to_feat,time)
    print('DONE!')
    print('Shape of array: ',cat_arr.shape)
    
    print('Initializing matrix for continuous variables...')
    cont_arr,cols_to_feat = format_long_data_cont(df3,cols_to_feat,time)
    print('DONE!')
    print('Shape of array: ',cont_arr.shape)
    
    print('Initializing matrix for medicinal data...')
    med_arr,cols_to_feat = format_med_data(df3,df4,cols_to_feat,time)
    print('DONE!')
    print('Shape of array: ',med_arr.shape)
    
    print('Initializing matrix for baseline data...')
    
    complete_data = np.concatenate((cat_arr,cont_arr,med_arr),axis=1)
    print('1. Shape of final array: ',complete_data.shape)
    
    complete_data,cols_to_feat = format_baseline_data(complete_data,df1,df3,cols_to_feat,time)
    print('5. Shape of final array: ',complete_data.shape)
    print('DONE!')
    
    #REST OF THE CATEGORICAL VARIABLES
    print('Initializing matrix for rest of the categorical variables...')
    li = ['alert','unresponsive','sedated','oriented','invasive','noninvasive','hfnc','dialysis']

    df3 = create_newdf3(df3,DF,time)
    temp = df3[['id']+li]
    for col in li:
        arr = np.zeros(len(df3['id'].unique()))
        i = 0
        for id in df3['id'].unique():
            df = temp[temp['id'] == id]
            df = df.fillna(0)
            if ((df[col]).sum() != 0):
                arr[i] = 1
            i += 1
        complete_data = np.c_[complete_data, arr]
        
    i = len(cols_to_feat)
    for j in range(i,i+8):
        cols_to_feat[j] = li[j-i]
        
    print('6. Shape of final array: ',complete_data.shape)
    
    return complete_data,cols_to_feat


def create_newdf3(df3,df,time):
    new_df3 = pd.DataFrame(columns=df3.columns)
    ids = sorted(list(df['id_x'].unique()))
    print('Creating new longitudinal dataframe ... ')
    for id in tqdm(ids):
        ubound = float(df[df['id_x'] == id]['dataCutoffTime'])
        lbound = 0.0
        if ubound > time:
            lbound = ubound - time

        if id not in df3['id'].unique():
            continue

        temp1 = df3[df3['id'] == id]['time'] <= ubound 
        temp2 = df3[df3['id'] == id]['time'] >= lbound
        temp = temp1 & temp2
        times = df3[df3['id'] == id]['time'][temp.values]
        app = df3[df3['id'] == id]
        app = app[app['time'].isin(list(times.values))]
        new_df3 = new_df3.append(app)
    new_df3 = new_df3.sort_values(['id'])
    return new_df3


def create_newdf4(df4,df,time):
    new_df4 = pd.DataFrame(columns=df4.columns)
    ids = sorted(list(df['id_x'].unique()))
    print('Creating new medication dataframe ... ')
    for id in tqdm(ids):
        ubound = float(df[df['id_x'] == id]['dataCutoffTime'])
        lbound = 0.0
        if ubound > time:
            lbound = ubound - time
        
        if id not in df4['id'].unique():
            continue

        temp1 = df4[df4['id'] == id]['time'] <= ubound 
        temp2 = df4[df4['id'] == id]['time'] >= lbound
        temp = temp1 & temp2
        times = df4[df4['id'] == id]['time'][temp.values]
        app = df4[df4['id'] == id]
        app = app[app['time'].isin(list(times.values))]
        new_df4 = new_df4.append(app)
    new_df4 = new_df4.sort_values(['id'])
    return new_df4


def generate(time):
    print('Reading datasets...')

    df1 = (pd.read_csv('/data1/dzeiberg/ARDS/2016-final/baseline-data.csv')).sort_values(['id'])
    df2 = (pd.read_csv('/data1/dzeiberg/ARDS/2016-final/review-data.csv')).sort_values(['id'])
    df3 = (pd.read_csv('/data1/dzeiberg/ARDS/2016-final/longitudinal-data.csv',low_memory=False)).sort_values(['id'])
    df4 = (pd.read_csv('/data1/dzeiberg/ARDS/2016-final/med-data.csv')).sort_values(['id'])

    print('Completed!')
    
    all_ids = sorted(df2['id'].unique())
    temp = df2[df2['ards_time'] <= time]['id'].unique()
    li = []
    for id in temp:
        if (mode(df2[df2['id'] == id]['ards_pt'])[0][0] == 0):
            li.append(id)
    censored = [x for x in temp if not x in li]
    ids = sorted([x for x in all_ids if not x in censored])
    ### For eligible ###
    ids = sorted(list(df1.dropna(subset=['elig_time_min'])['id'].unique()))

    df1 = (df1[df1['id'].isin(ids)]).sort_values(['id'])
    df3 = (df3[df3['id'].isin(ids)]).sort_values(['id'])
    df4 = (df4[df4['id'].isin(ids)]).sort_values(['id'])

    X,dic = format_design_matrix(df1,df2,df3,df4,time)
    y = format_output(df2,ids)

    np.save(getFeaturesPath(),X)
    np.save(getLabelsPath(),y)
    np.save('dict_test.npy',dic)
    return X,y,dic


def main():
    time = 6*60
    X1,y1,dic1 = generate(time)
    print("Shape of final labels arrray: ", y1.shape)
    print("Number of positive patients: ", sum(y1))
    print("Feature dictionary: ", dic1)


if __name__ == "__main__":
    main()