# Mind Wandering Detection on SART

* Prepare Dataset
    * Path: ``./data/raw``
    * user01-user60 + label + eprime
    
* Subjects: 12 people => [6,7,8,13,14,18,26,31,32,40,42,43]

* Script
    ``bash segmentation.sh``
    ``bash preprocess.sh``
    ``bash train.sh``

* Segmentation [L-label, D-data, obj-objective, sub-subjective]
    * Objective Experiment [Before/After response 10/20]

    ``python src/segmentation/seg_L-obj_D-response.py [seg_pkl_name] --window=before --time=10 --plot=False --normalize=False``

    * Subjective Experiment [Before/After response 10/20]

    ``python src/segmentation/seg_L-sub_D-response.py [seg_pkl_name] --window=before --time=10 --plot=False --normalize=False``

    * Subjective Experiment [Before probe 10/20/30/40/all]

    ``python src/segmentation/seg_L-sub_D-probe.py [seg_pkl_name] --time=10 --plot=False --normalize=False``

* Data Preprocessing + Feature extraction

`` python src/preprocess/gsr_extraction.py ./data/seg_data/[seg_pkl_name] [feature_csv_name]``

* Train XGB Binary Class

``python src/train/trainGSR_xgb_binary.py [feature_csv_name]``
