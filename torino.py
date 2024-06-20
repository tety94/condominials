import geopandas as gpd
import numpy as np
import pandas as pd
from mysql.connector import (connection)
import matplotlib.pyplot as plt
from utilities import generate_closest_array
import random
from collections import Counter
from scipy import stats
from get_data import get_data

USE_QUARTERS = True

CRS_PIEMONTE = 32632
CRS_PIEMONTE_STRING = 'EPSG:32632'
CRS_MARCATORE_STRING = 'EPSG:4326'
CRS_MARCATORE = 4326

N_ITERAZIONS = 10000

################
### GET DATA ###
################
# This method returns two GeoDataFrames. The first contains the double or triple buildings, and the second contains all the historical addresses
buildings_gdf, total_historical_gdf = get_data()

#########################################
### INSTANTIATE THE GDF FOR ARPA DATA ###
#########################################
BUILDING_TORINO_PATH = "" #path of the shape file with Torino buildings

torino_buildings_gdf = gpd.read_file(BUILDING_TORINO_PATH)
torino_buildings_gdf.set_crs(epsg=CRS_PIEMONTE, inplace=True, allow_override=True)
torino_buildings_gdf.to_crs(epsg=CRS_PIEMONTE, inplace=True)

torino_buildings_gdf['POP_INT'] = np.NaN
torino_buildings_gdf['POP_INT'] = round(torino_buildings_gdf.POP).astype(int)
torino_buildings_gdf.ID_EDIF = round(torino_buildings_gdf.ID_EDIF).astype(int)

torino_buildings_gdf = torino_buildings_gdf[torino_buildings_gdf.POP_INT > 0]

########################################
### INSTANTIATE THE GDF FOR QUARTERS ###
########################################
QUARTERS_PATH = "" #path of the shapefile with the boundaries of Torino's quarters
quarters_gdf = gpd.read_file(QUARTERS_PATH, encoding='latin-1')
quarters_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
quarters_gdf.to_crs(epsg=CRS_PIEMONTE, inplace=True)

#####################################
### ADD BUILDINGS IN THE QUARTERS ###
#####################################
if USE_QUARTERS:
    quarters_gdf['double_total_buildings'] = 0
    quarters_gdf['triple_total_buildings'] = 0

    bb = total_historical_gdf[(total_historical_gdf['n_after_removing_mutation_and_familial'] == 2) & (
        total_historical_gdf['arpa_building_id'].notna())]['arpa_building_id'].tolist()
    bb = [int(x) for x in bb]
    ov_d = quarters_gdf.overlay(torino_buildings_gdf[torino_buildings_gdf.ID_EDIF.isin(bb)], keep_geom_type=True)
    v_d = ov_d.groupby(by='ID_QUART').ID_QUART.count()
    bb = total_historical_gdf[(total_historical_gdf['n_after_removing_mutation_and_familial'] >= 3) & (
        total_historical_gdf['arpa_building_id'].notna())]['arpa_building_id'].tolist()
    bb = [int(x) for x in bb]
    ov_t = quarters_gdf.overlay(torino_buildings_gdf[torino_buildings_gdf.ID_EDIF.isin(bb)], keep_geom_type=True)
    v_t = ov_t.groupby(by='ID_QUART').ID_QUART.count()

    for index, q in quarters_gdf.iterrows():
        if q.ID_QUART in list(v_d.keys()):
            quarters_gdf.at[index, 'double_total_buildings'] = v_d[q.ID_QUART]
        if q.ID_QUART in list(v_t.keys()):
            quarters_gdf.at[index, 'triple_total_buildings'] = v_t[q.ID_QUART]

##################
### MONTACARLO ###
##################
def monte_carlo(historical, buildings_gdf, quarters_gdf=np.NaN, use_quarters=False):
    # create array containing building id as many times as many people live in it
    buildings_array = []
    for index, value in buildings_gdf.iterrows():
        l = [value.ID_EDIF] * value.POP_INT
        buildings_array.extend(l)

    # calculating how many times patients have moved house in their lifetime
    n_patients = len(historical['codals_parals'].unique())
    mean_change_house = len(historical['codals_parals']) / n_patients

    # create array that allows you to create same distribution of home changes
    print(f"There were {n_patients} who changed homes on average {mean_change_house} times")
    n_change_house = generate_closest_array(mean_change_house, 5)

    total_d_buildings = []
    total_t_buildings = []
    patients_array = np.arange(n_patients)

    # create empty arrays
    if use_quarters:
        quarters_gdf['predicted_d_buildings_mean'] = 0
        quarters_gdf['predicted_t_buildings_mean'] = 0
        quarters_gdf['predicted_d_buildings_std'] = 0
        quarters_gdf['predicted_t_buildings_std'] = 0

        d_iterations = {}
        for index, q in quarters_gdf.iterrows():
            d_iterations[q.ID_QUART] = []

        t_iterations = {}
        for index, q in quarters_gdf.iterrows():
            t_iterations[q.ID_QUART] = []

    for iteration in np.arange(N_ITERAZIONS):
        # start MonteCarlo iterarion
        # create array with the random choosen buildings
        buildings = []
        # select random buildings for each patients
        for patients in patients_array:
            change_house_times = random.sample(n_change_house, 1)[0]
            patient_buildings = random.sample(buildings_array, change_house_times)
            buildings = buildings + patient_buildings
        c = Counter(buildings).items()
        total_t_buildings.append(len(list({k: v for k, v in c if v >= 3})))
        total_d_buildings.append(len(list({k: v for k, v in c if v == 2})))

        # analysis for the quarters
        if use_quarters:
            bb = list({k: v for k, v in c if v == 2})
            ov_d = quarters_gdf.overlay(buildings_gdf[buildings_gdf.ID_EDIF.isin(bb)], keep_geom_type=True)
            v_d = ov_d.groupby(by='ID_QUART').ID_QUART.count()
            bb = list({k: v for k, v in c if v == 3})
            ov_t = quarters_gdf.overlay(buildings_gdf[buildings_gdf.ID_EDIF.isin(bb)], keep_geom_type=True)
            v_t = ov_t.groupby(by='ID_QUART').ID_QUART.count()

            for index, q in quarters_gdf.iterrows():
                if q.ID_QUART in list(v_d.keys()):
                    d_iterations[q.ID_QUART].append(v_d[q.ID_QUART])
                else:
                    d_iterations[q.ID_QUART].append(0)
                if q.ID_QUART in list(v_t.keys()):
                    t_iterations[q.ID_QUART].append(v_t[q.ID_QUART])
                else:
                    t_iterations[q.ID_QUART].append(0)

    # end MonteCarlo iterarion

    for index, q in quarters_gdf.iterrows():
        quarters_gdf.loc[quarters_gdf['ID_QUART'] == q.ID_QUART, 'predicted_d_buildings_mean'] = np.mean(
            d_iterations[q.ID_QUART])
        quarters_gdf.loc[quarters_gdf['ID_QUART'] == q.ID_QUART, 'predicted_d_buildings_std'] = np.std(
            d_iterations[q.ID_QUART])

        quarters_gdf.loc[quarters_gdf['ID_QUART'] == q.ID_QUART, 'predicted_t_buildings_mean'] = np.mean(
            t_iterations[q.ID_QUART])
        quarters_gdf.loc[quarters_gdf['ID_QUART'] == q.ID_QUART, 'predicted_t_buildings_std'] = np.std(
            t_iterations[q.ID_QUART])

    monte_carlo_df = pd.DataFrame({
        'total_d_buildings': np.array(total_d_buildings),
        'total_t_buildings': np.array(total_t_buildings),
    })

    return monte_carlo_df, quarters_gdf


monte_carlo_df, quarters_gdf = monte_carlo(total_historical_gdf, torino_buildings_gdf, quarters_gdf,
                                           use_quarters=USE_QUARTERS)

monte_carlo_df.to_csv('montecarloResults.csv', sep=';')

quarters_gdf['predicted_d_buildings_mean'] = np.round(quarters_gdf['predicted_d_buildings_mean'], 2)
quarters_gdf['predicted_t_buildings_mean'] = np.round(quarters_gdf['predicted_t_buildings_mean'], 2)
quarters_gdf['predicted_d_buildings_std'] = np.round(quarters_gdf['predicted_d_buildings_std'], 2)
quarters_gdf['predicted_t_buildings_std'] = np.round(quarters_gdf['predicted_t_buildings_std'], 2)

quarters_gdf['ratio_d'] = np.round(quarters_gdf['double_total_buildings'] / quarters_gdf['predicted_d_buildings_mean'],
                                   2)
quarters_gdf['ratio_t'] = np.round(quarters_gdf['triple_total_buildings'] / quarters_gdf['predicted_t_buildings_mean'],
                                   2)

quarters_gdf['z_d'] = np.round(
    (quarters_gdf['double_total_buildings'] - quarters_gdf['predicted_d_buildings_mean']) / quarters_gdf[
        'predicted_d_buildings_std'], 5)
quarters_gdf['pval_d'] = np.round(stats.norm.sf(abs(quarters_gdf['z_d'])) * 2, 5)
quarters_gdf['z_t'] = np.round(
    (quarters_gdf['triple_total_buildings'] - quarters_gdf['predicted_t_buildings_mean']) / quarters_gdf[
        'predicted_t_buildings_std'], 5)
quarters_gdf['pval_t'] = np.round(stats.norm.sf(abs(quarters_gdf['z_t'])) * 2, 5)

pd.DataFrame(quarters_gdf.drop(columns='geometry')).to_csv('quartersResults.csv', sep=';', decimal=',')

###############
### RESULTS ###
###############
mu = monte_carlo_df.total_d_buildings.mean()
sigma = monte_carlo_df.total_d_buildings.std()

value = len(buildings_gdf[buildings_gdf['n_after_removing_mutation_and_familial'] == 2])
z = (value - mu) / sigma
p_values = stats.norm.sf(abs(z)) * 2  # twosided

print(f"In Torino there are {value} condominiums that are ONLY double.")
print(
    f"Monte Carlo gives an average of {monte_carlo_df.total_d_buildings.mean()} double buildings with STD {monte_carlo_df.total_d_buildings.std()}")
print(f"This value is away from the mean {z} deviations and pvalue of {p_values}")

mu = monte_carlo_df.buildings_t_totali.mean()
sigma = monte_carlo_df.buildings_t_totali.std()

value = len(buildings_gdf[buildings_gdf['n_after_removing_mutation_and_familial'] >= 3])
z = (value - mu) / sigma
p_values = stats.norm.sf(abs(z)) * 2  # twosided

print(f"In Torino there are {value} condominiums that are ONLY triple.")
print(
    f"Monte Carlo gives an average of {monte_carlo_df.total_d_buildings.mean()} double buildings with STD {monte_carlo_df.total_d_buildings.std()}")
print(f"This value is away from the mean {z} deviations and pvalue of {p_values}")
