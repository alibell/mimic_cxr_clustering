import pandas as pd
from .dataset import imageDataset


# Ensemble des scripts de pre-processing

def get_data(data_folder="./data"):
    """
        Parameters
        ----------
        data_folder: str, folder containing the date

        Return
        ------
        - tuple containing :
            > an image dataset object
            > a pandas dataframe of metadata
        - dataframe of labels
    """

    # Creating an UID
    df_expert = pd.read_csv(f"{data_folder}/mimic-cxr-2.0.0-chexpert.csv.gz")
    df_metadata = pd.read_csv(f"{data_folder}/mimic-cxr-2.0.0-metadata.csv.gz")

    df_metadata = df_metadata.reset_index() \
        .rename(columns = {"index":"uid"}) \
        .drop(columns = ["Unnamed: 0"])

    df_expert = df_expert \
        .drop(columns = ["Unnamed: 0"]) \
        .join(df_metadata[["study_id","uid"]].set_index("study_id"), on = "study_id") \
        .set_index("uid") \
        .sort_index() \
        .reset_index(drop=False)
        
    df_expert["text_label"] = df_expert.iloc[:,3:].apply(
        lambda x: ",".join(x.dropna()[x.dropna() == 1].index.tolist())
    , axis=1)

    # Getting reports
    subject_series = df_metadata["subject_id"].astype("str")
    study_series = df_metadata["study_id"].astype("str")
    df_expert["report"] = (data_folder+"/files/p"+subject_series.str.slice(0,2)+"/p"+subject_series+"/s"+study_series+".txt") \
        .apply(lambda x: open(x, "r").read())

    # Getting reports sub-sections
    report = pd.pivot_table(
    df_expert[["uid", "report"]] \
            .assign(report=lambda x: x["report"].str.replace("(INDICATION:|COMPARISON:|FINDINGS:|IMPRESSION:?|REASON FOR EXAM((?i)INATION)?:)|HISTORY:|TECHNIQUE:","#SPLIT_LEFT#\\1#SPLIT_RIGHT#", regex=True) \
                        .str.split("#SPLIT_LEFT#")
            ).explode("report").set_index("uid")["report"] \
            .str.split("#SPLIT_RIGHT#", expand=True) \
            .assign(
                text=lambda x: x[1].str.replace("\n", " ").str.strip(),
                category=lambda x: x[0].str.replace(":","").str.lower().str.replace(" +", "_", regex=True)
            ).reset_index(), 
            values="text",
            columns="category",
            index="uid",
            aggfunc=lambda x: x
    ).fillna("")
    report.columns = ["report_"+x if x != "" else "report_extra" for x in report.columns]

    df_expert = pd.merge(
        df_expert,
        report,
        left_on="uid",
        right_on="uid",
        how="left"
    )
    df_expert[report.columns[1:]].fillna("")

    # Getting image loader

    ## Getting image path dict
    images_paths = df_metadata[["uid", "subject_id", "dicom_id", "study_id"]] \
                    .reset_index(drop=True)

    images_paths["subject_id_str"] = images_paths["subject_id"].astype("str")
    images_paths["study_id_str"] = images_paths["study_id"].astype("str")

    images_paths["path"] = f"{data_folder}/./files/p"+images_paths["subject_id_str"].str.slice(0,2)+"/p"+ \
        +images_paths["subject_id_str"]+"/s"+ \
        +images_paths["study_id_str"]+"/"+ \
        +images_paths["dicom_id"]+".jpg"

    images_paths = images_paths[["uid","path"]].set_index("uid")["path"] \
            .to_dict()

    ## Loading images
    images_dataset = imageDataset(images_paths)

    return (images_dataset, df_metadata), df_expert