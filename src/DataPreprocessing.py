import pandas as pd
import os.path

def merge_forcage_piezo(forcage_file, merge_file ='dataset/static_attributes/piezo_characs.csv', output_file = None):
    df_forcage =  pd.read_csv('dataset/time_series/forçages/' + forcage_file + '.txt', 
                  sep=';', 
                  skiprows=85,
                  comment='#')
    df_forcage.columns = df_forcage.columns.str.strip()
    df_piezo_charachs =  pd.read_csv(merge_file,sep=';',encoding='latin-1')
    
    bss_id = df_piezo_charachs[df_piezo_charachs['limnit'] == forcage_file ]['BSS_ID'].values[0]
    
    df_piezo = pd.read_csv("dataset/time_series/piezos/"+bss_id+".csv")
    
    df_piezo.head()

    df_piezo["date_mesure"] = (
        pd.to_datetime(df_piezo["date_mesure"])
        .dt.strftime("%Y%m%d")
        .astype("Int64")
    )

    df_piezo['Date'] = df_piezo['date_mesure'].astype('int64')
    #garder que les colonnes 'date_mesure' et 'niveau_nappe_eau'
    df_piezo = df_piezo[['Date', 'niveau_nappe_eau']]
    #join df_forcage et df_piezo sur la colonne 'date'
    df_merged = pd.merge(df_forcage, df_piezo, on='Date', how='inner')

    if(output_file != None):
        df_merged.to_csv(output_file, index=False)
        
    return df_merged

# Return the mean level of water for the last 4 year associated to this bss id
def water_level_mean(bss_id):
    if type(bss_id) != str:
        bss_id = bss_id['BSS_ID']
    if os.path.isfile("dataset/time_series/piezos/"+str(bss_id)+".csv"):
        df_piezo = pd.read_csv("dataset/time_series/piezos/"+str(bss_id)+".csv")
        df_piezo.head()
        df_piezo["date_mesure"] = (
            pd.to_datetime(df_piezo["date_mesure"])
            .dt.strftime("%Y%m%d")
            .astype("Int64")
        )
        date_max = df_piezo["date_mesure"].astype('int64').max()
        return df_piezo[df_piezo["date_mesure"] > date_max - 40000]["niveau_nappe_eau"].mean()
    return pd.NaT

#Extract usefull data from piezo_characs with mean water level
def get_attributes():
    raw_df_piezo_characs = pd.read_csv('dataset/static_attributes/piezo_characs.csv', sep=';', encoding='latin-1')
    df_piezo_characs = raw_df_piezo_characs[["BSS_ID", "H", "XL93", "YL93", "formation", "état", "nature", "milieu", "thème", "limnit", "rr_mean", "rr_p", "etp_mean", "stress_p"]].copy()
        
    # Associate the water level to the data
    df_piezo_characs["water_level_mean"] = df_piezo_characs.apply(water_level_mean, axis=1)

    # Supress row with empty values
    df_piezo_characs.dropna(inplace=True)
    df_piezo_characs.reset_index(drop=True, inplace=True)
    
    # Drop the non real columns 
    df_piezo_characs.drop(columns=['BSS_ID', 'limnit', "état"], inplace=True)
    # Supress columns with only one values 
    df_len = len(df_piezo_characs.index)
    columns = []
    for col in list(df_piezo_characs.columns):
        if df_piezo_characs[col].value_counts(ascending=True).values[0] == df_len :
            columns.append(col)
    df_piezo_characs.drop(columns=columns, inplace=True)

    return df_piezo_characs


# get all the statics attributes of all well with the mean water level
def get_static_attributes():
    #Extract usefull data from piezo_characs
    raw_df_piezo_characs = pd.read_csv('dataset/static_attributes/piezo_characs.csv', sep=';', encoding='latin-1')
    df_piezo_characs = raw_df_piezo_characs[["BSS_ID", "H", "XL93", "YL93", "formation", "état", "nature", "milieu", "thème", "limnit", "rr_mean", "rr_p", "etp_mean", "stress_p"]]
    
    df_static_attributes_geo =  pd.read_csv('dataset/static_attributes/geology_attributes_bh.csv', sep=';')
    df_static_attributes_hydrogeo =  pd.read_csv('dataset/static_attributes/hydrogeology_attributes_bh.csv', sep=';')
    #df_static_attributes_hydrolog =  pd.read_csv('dataset/static_attributes/CAMELS_FR_hydrological_signatures.csv', sep=';')
    #df_static_attributes_topo =  pd.read_csv('dataset/static_attributes/CAMELS_FR_topography_general_attributes.csv', sep=';')

    #Extract usefull data from CAMELS_FR_soil_general_attributes
    raw_df_g =  pd.read_csv('dataset/static_attributes/soil_general_attributes_bh.csv', sep=';')
    df_static_attributes_general = raw_df_g[["sta_code_h3"]].copy()
    for sol_agg_level in ["no", "top_subsoil", "topsoil"] :
        for sol_stat in ["mean", "skewness", "na_pct"]:
            df_tmp = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]
            if sol_stat == "na_pct":
                if sol_stat != "no":
                    df_static_attributes_general['sol_conductivity_'+sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_conductivity"]
                    if sol_stat == "top_subsoil":
                        df_static_attributes_general['sol_tawc_inrae_'+sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_tawc_inrae"]
            else:
                if sol_agg_level == "no":
                    df_static_attributes_general['sol_depth_to_root_'+sol_stat] = df_tmp["sol_depth_to_root"]
                    df_static_attributes_general['sol_depth_to_bedrock_'+sol_stat] = df_tmp["sol_depth_to_bedrock"]
                else :
                    df_static_attributes_general['sol_clay_'        +sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_clay"]
                    df_static_attributes_general['sol_sand_'        +sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_sand"]
                    df_static_attributes_general['sol_silt_'        +sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_silt"]
                    df_static_attributes_general['sol_oc_'          +sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_oc"]
                    df_static_attributes_general['sol_bd_'          +sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_bd"]
                    df_static_attributes_general['sol_gravel_'      +sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_clay"]
                    df_static_attributes_general['sol_tawc_'        +sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_tawc"]
                    df_static_attributes_general['sol_conductivity_'+sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_conductivity"]
                    if sol_agg_level == "top_subsoil":
                        df_static_attributes_general['sol_tawc_inrae_'+sol_agg_level+"_"+sol_stat] = raw_df_g[(raw_df_g["sol_stat"] == sol_stat) & (raw_df_g["sol_agg_level"] == sol_agg_level)]["sol_tawc_inrae"]
    df_static_attributes_general = df_static_attributes_general.groupby('sta_code_h3').sum()

    # Merge all
    df_all_attributes = pd.merge(df_piezo_characs, df_static_attributes_geo, left_on="limnit", right_on="sta_code_h3", how="left")
    df_all_attributes = pd.merge(df_all_attributes, df_static_attributes_hydrogeo, on="sta_code_h3", how="left")
    #df_all_attributes = pd.merge(df_all_attributes, df_static_attributes_hydrolog, on="sta_code_h3", how="left")
    #df_all_attributes = pd.merge(df_all_attributes, df_static_attributes_topo, on="sta_code_h3", how="left")
    df_all_attributes = pd.merge(df_all_attributes, df_static_attributes_general, on="sta_code_h3", how="left")
    df_all_attributes.dropna(inplace=True)

    # Associate the water level to the data
    df_all_attributes["water_level_mean"] = df_all_attributes.apply(water_level_mean, axis=1)

    # Supress row with empty values
    df_all_attributes.dropna(inplace=True)
    df_all_attributes.reset_index(drop=True, inplace=True)

    # Drop the non real columns 
    df_all_attributes.drop(columns=['BSS_ID', 'limnit', 'état', 'sta_code_h3', 'geo_dom_class'], inplace=True)

    # Supress columns with only one values 
    df_len = len(df_all_attributes.index)
    columns = []
    for col in list(df_all_attributes.columns):
        if df_all_attributes[col].value_counts(ascending=True).values[0] == df_len :
            columns.append(col)
    df_all_attributes.drop(columns=columns, inplace=True)

    return df_all_attributes