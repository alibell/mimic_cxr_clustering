from cv2 import split
import pandas as pd
import numpy as np
import zipfile
from .dataset import imageDataset


# Ensemble des scripts de pre-processing
def read_annotations(path):
    annotations = pd.read_json(
        zipfile.ZipFile(path).open("all.json")
    )
    annotations["n_label"] = annotations["label"].apply(lambda x: len(x))
    labels = annotations["label"].explode().dropna().unique()
    labels_colname = [f"annotation_{x}" for x in labels]

    for label, label_colname in zip(labels, labels_colname):
        annotations[label_colname] = annotations["label"].apply(lambda x: 1*(label in x))
    annotations.loc[(annotations["n_label"] == 0), labels_colname] = np.nan
    annotations = annotations.dropna()

    return annotations

def get_imageDataset(df_metadata, data_folder):
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

    images_dataset = imageDataset(images_paths)

    return images_dataset


def get_data(data_folder="./data", annotations_path=None, split_annotation=True):
    """
        Parameters
        ----------
        data_folder: str, folder containing the date
        annotations_path: str, path to the annotations file
        split_annotation: boolean, if true the dataset is splitted between data with and without annotations

        Return
        ------
        - tuple containing :
            > an image dataset object
            > a pandas dataframe of metadata
        - dataframe of labels

        If split_annotation is set to True, the function output 2 tuples and 2 dataframes
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

    # Adding annotations
    if annotations_path is not None:
        annotations = read_annotations(annotations_path)
        annotations_with_study_id = pd.merge(
            annotations,
            df_expert[["uid", "study_id"]],
            left_on="uid",
            right_on="uid",
            how="inner"
        )

        df_expert = pd.merge(
            df_expert,
            annotations_with_study_id.drop(columns=["uid"]),
            left_on="study_id",
            right_on="study_id",
            how="left"
        )

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

    if annotations_path is not None and split_annotation:
        mask = (df_expert[df_expert.columns[df_expert.columns.str.contains("annotation")]].isna()).sum(axis=1) == 0
        df_expert_annotation = df_expert[mask].reset_index(drop=True)
        df_expert_no_annotation = df_expert[mask == False].reset_index(drop=True)
        df_metadata_annotation = df_metadata[mask].reset_index(drop=True)
        df_metadata_no_annotation = df_metadata[mask == False].reset_index(drop=True)
        images_dataset_annotation = get_imageDataset(df_metadata=df_metadata_annotation, data_folder=data_folder)
        images_dataset_no_annotation = get_imageDataset(df_metadata=df_metadata_no_annotation, data_folder=data_folder)

        return (images_dataset_no_annotation, df_metadata_no_annotation), df_expert_no_annotation, (images_dataset_annotation, df_metadata_annotation), df_expert_annotation
    else:
        images_dataset = get_imageDataset(df_metadata=df_metadata, data_folder=data_folder)

        return (images_dataset, df_metadata), df_expert