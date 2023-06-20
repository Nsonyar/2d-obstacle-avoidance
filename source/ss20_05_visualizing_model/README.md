# ss20_05_visualizing_model
Here a model or a dataset can be used to visualize it as an .avi video. The
videos will be stored in subfolders with a specific identifier in order to
relate them to models or datasets they are made from.

Following command is used to create a video based on a given model and a 
specific bag which can be set in the script:
```bash
python3 visualizing_model.py
```
The script can also be launched with parameters as follows:
```bash
python3 visualizing_model.py [-bi BAGINDEX]
```
- **[-bi BAGINDEX]** -> index of the bag to generate the video from