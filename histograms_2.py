import pandas as pd
import numpy as np


# -----------------------------------------------------------
# Read excel file
# -----------------------------------------------------------
df = pd.ExcelFile('Hacker_Data_BaseN.xlsx').parse('Patterns')

# -----------------------------------------------------------
# Country IDS
# -----------------------------------------------------------
df['Country'] = df['Country'].str.upper()
country_unique = df['Country'].drop_duplicates().tolist()
country_sort = np.sort(country_unique)
# Save Country
with open("COUNTRY_.txt", "w") as file:
    for element in country_sort:
        file.write(element + "\n")
print('----->>> Save COUNTRIES successfully !!')
# -----------------------------------------------------------
# Tool IDS
# -----------------------------------------------------------
df['Tool Attack'] = df['Tool Attack'].str.upper()
tool_unique = df['Tool Attack'].drop_duplicates().tolist()
tool_sort = np.sort(tool_unique)
# Save Tools
with open("TOOL.txt", "w") as file:
    for element in tool_sort:
        file.write(element + "\n")
print('----->>> Save TOOLS successfully !!')
# -----------------------------------------------------------
# Attacks IDS
# -----------------------------------------------------------
df['Type Attack'] = df['Type Attack'].str.upper()
type_unique = df['Type Attack'].drop_duplicates().tolist()
type_sort = np.sort(type_unique)
# Save Attacks
with open("ATTACK.txt", "w") as file:
    for element in type_sort:
        file.write(element + "\n")
print('----->>> Save ATTACKS successfully !!')


# Personality test

# -----------------------------------------------------------
# PERSONALITY IDS
# -----------------------------------------------------------
df['Personality'] = df['Personality'].str.upper()
person_unique = df['Personality'].drop_duplicates().tolist()
person_sort = np.sort(person_unique)
# Save personality
with open("PERSONALITY.txt", "w") as file:
    for element in person_sort:
        file.write(element + "\n")
print('----->>> Save PERSONALITY successfully !!')
# -----------------------------------------------------------
# TRAIT IDS
# -----------------------------------------------------------
df['Trait'] = df['Trait'].str.upper()
trait_unique = df['Trait'].drop_duplicates().tolist()
trait_sort = np.sort(trait_unique)
# Save trait
with open("TRAIT.txt", "w") as file:
    for element in trait_sort:
        file.write(element + "\n")
print('----->>> Save TRAIT successfully !!')
# -----------------------------------------------------------
# MOTIVATION IDS
# -----------------------------------------------------------
df['Motivation'] = df['Motivation'].str.upper()
motiv_unique = df['Motivation'].drop_duplicates().tolist()
motiv_sort = np.sort(motiv_unique)
# Save motivation
with open("MOTIVATION.txt", "w") as file:
    for element in motiv_sort:
        file.write(element + "\n")
print('----->>> Save MOTIVATION successfully !!')