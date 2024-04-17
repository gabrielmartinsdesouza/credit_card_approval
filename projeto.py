# %%
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# %%
df_credit_record = pd.read_csv("datasets/credit_record.csv")
df_application_record = pd.read_csv("datasets/application_record.csv")

# %%
df_application_record.shape


# %%
df_application_record.head()

# %%
df_application_record.info()


# %%
# Verificando linhas duplicadas
df_application_record.duplicated().sum()

# %%
df_application_record["ID"].duplicated().sum()

# %%
47/df_application_record.shape[0]

# %%
# Removendo os IDs duplicados.
df_application_record = df_application_record[df_application_record["ID"].duplicated() == False].copy()

# %%
list(df_application_record)

# %%
cat_attribs = [
    'ID', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'FLAG_MOBIL',
    'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE']
num_attribs = [
 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']

# %%
df_ar_cat = df_application_record[cat_attribs]
df_ar_num = df_application_record[num_attribs]

# %%
df_ar_train, df_ar_test = train_test_split(df_application_record,
                                           test_size=0.2,
                                           random_state=42)

# %%
df_ar_cat = df_ar_train[cat_attribs]
df_ar_num = df_ar_train[num_attribs]

# %%
df_ar_cat.info()

# %%

for attribs in list(df_ar_cat)[1:]:
    categories = df_ar_cat[attribs].value_counts().index
    values = df_ar_cat[attribs].value_counts()
    
    fig, ax = plt.subplots(figsize=(10,5))
    
    bar = ax.bar(categories, values)
    ax.set_title(attribs, fontsize=15)
    ax.bar_label(ax.containers[0], label_type='edge', fontsize=12)

    plt.xticks(fontsize=12, rotation=45)
    plt.show()

# %%
df_ar_cat['FLAG_MOBIL'].value_counts()

# %%
df_ar_cat["NAME_HOUSING_TYPE"].value_counts().index

# %%
df_ar_cat.head()

# %%
pd.options.display.float_format = "{:.2f}".format  # Alterando o formato de exibição do pandas para evitar notação científica.
df_ar_num.describe()

# %%
df_ar_cat["ID"].value_counts()

# %%
df_ar_train.shape

# %%
df_ar_test.shape

# %%
87702/438557