## Data Directory

A directory that will be mapped to `${AMLT_DATA_DIR}` on Amulet. Suupose its local path is `${LOCAL_DATA_DIR}`, then the data directory has the following layout:

```
> tree -d -L 1 ${LOCAL_DATA_DIR}
${LOCAL_DATA_DIR}
|-- annos
|-- data
|-- models
`-- snapshots
```

Each folder stores files described as follows (suupose a dataset has name `${DATASET}`):

- Directory **annos**: store dataset annotation file `${DATASET}.b.json` (base file) and `${DATASET}.g.json` (pseudo annotation file).
- Directory **data**: store dataset images in a folder `${DATASET}`. All frames are in format `%06d.jpg`.
- Directory **models**: only two pretrained models `ssd.pth` and `r101.pth`.
- Directory **snapshots**: store inference results `${DATASET}.r.pkl` from the expert model.

Download annotations and snapshots of `detrac_1~detrac_6` and `ut_1~ut_20` from Azure Storage and put them under corresponding directly described as above.

## Data Preprocessing

### Generate DETRAC dataset

Place DETRAC train and test dataset in the **same** folder. Suppose its name is `${DETRAC_DIR}`, then each trace has a path of format `${DETRAC_DIR}/MVI_*/`. There should be 100 traces in total. Run:

```
python tools/util/gen_detrac.py -i ${DETRAC_DIR} -o ${LOCAL_DATA_DIR}/data
```

There should be 6 traces `detrac_1~detrac_6` generated under `${LOCAL_DATA_DIR}/data`.

### Generate Urban Traffic dataset

Place UT dataset under a folder `${UT_DIR}`. Traces from each location are placed under the same child directory, e.g., `${UT_DIR}/Bellevue_116th_NE12th/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4`. Run:

```
python tools/util/gen_ut.py -i ${UT_DIR} -o ${LOCAL_DATA_DIR}/data
```

There should be 20 datasets `ut_1~ut_20` generated under `${LOCAL_DATA_DIR}/data` (this may take quite long).
