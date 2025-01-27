# Grades Auto Filler
 Image Processing Grades Auto filler Project 🖼️


## Table of Contents

- <a href ="#overview">Overview</a>
- <a href ="#started">Get Started</a>
- <a href ="#graded_sheet">Graded Sheet Model</a>
  - <a href ="#graded_sheet_overview">Overview</a>
  - <a href ="#graded_sheet_results">Results</a>
- <a href ="#bubble_sheet">Bubble Sheet Model</a>
  - <a href ="#bubble_sheet_overview">Overview</a>
  - <a href ="#bubble_sheet_results">Results</a>
- <a href ="#video_demo">Demo Video</a>
- <a href ="#contributors">Contributors</a>
- <a href ="#license">License</a>


## <img  align= center width=40px height=40px src="https://media3.giphy.com/media/psneItdLMpWy36ejfA/source.gif"> Overview<a id = "overview"></a>
- Project Based on Image Processing Techniques
- We have 2 models
  - Graded Sheet Model
  - Bubble Sheet Model
- <a href="https://github.com/eslamwageh/Grades-auto-filler/blob/main/Grades%20autofiller%20%5BOptional%20Idea%5D.docx">Project Document</a>

## <img  align= center width=50px height=50px src="https://user-images.githubusercontent.com/72309546/215230425-03645465-e762-42ae-9772-947ca1b01401.png">Used Technolgies <a id = "Technolgies"></a>

<div>
<img height="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png" />

<img height="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png" />

<img height="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/OpenCV_Logo_with_text_svg_version.svg/800px-OpenCV_Logo_with_text_svg_version.svg.png" />
<img height="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/2560px-Scikit_learn_logo_small.svg.png" />
<img height="40" src="https://miro.medium.com/max/490/1*9Gbo-HOvajpbya5RsLN1uw.png" />
<img height="40" src="https://i0.wp.com/iot4beginners.com/wp-content/uploads/2020/04/65dc5834-de21-4e2e-bd4d-5e0c3c6994dd.jpg?fit=375%2C422&ssl=1" alt="GUI tool for python"/>
</div>

## <img  align= center width=40px height=40px src="https://cdn.pixabay.com/animation/2022/07/31/06/27/06-27-17-124_512.gif">Get Started <a id = "started"></a>

<ol>
<li>Clone the repository

<br>

```
git clone https://github.com/eslamwageh/Grades-auto-filler.git
```

</li>

<li>Install Packages

<br>

```
pip install -r requirements.txt
```

</li>

<li>Run the app

<br>

```
python GUI.py
```

</li>
</ol>

<h2 align=center > <img  align=center width=50px height=50px src="https://user-images.githubusercontent.com/72309546/214907431-b4e250f1-9b3a-4149-b7b4-bbd17b833b97.png">Graded Sheet Model <a id = "graded_sheet"></a>
</h2>


### <img  align= center width=30px height=30px src="https://media1.giphy.com/media/3ohzdWYbITsO2Y5rbi/giphy.gif?cid=6c09b9523ys3hxe1y9ueyo5ab7nzkbhv9oev797jvb5bz6rt&rid=giphy.gif&ct=s"> OverView<a id = "graded_sheet_overview"></a>

- It allows you to fill the grades electronically
- It handles skewing, orientation, and different lighting conditions
- Printed Student ID is detected using OCR and Features & Classifier
- Colomns and Rows are separated using Hough Transform Algorithm to detect table cells
- Written Symbols like ✓ & x are detect using HOG feature extractor and predicted using SVM Classifier
- Handwritten Code Values are detected using OCR and Features & Classifier

***

### <img  align= center width=50px height=50px src="https://cdn-icons-png.flaticon.com/512/1589/1589689.png"> Results<a id = "graded_sheet_results"></a>

<h4 align=center> <img  align= center width=20px height=20px src="https://media1.giphy.com/media/ZecwzuvmRrjOHsXNcI/giphy.gif?cid=6c09b9523btueuk8qe6usw2cnpb7qn8ki6evjwp62n2xiyi7&rid=giphy.gif&ct=s">Graded Sheet using features & classifer<a id = "results"></a>
</h4>


<table>
  <tr>
    <td width=40% valign="center"><img src="https://raw.githubusercontent.com/eslamwageh/Grades-auto-filler/refs/heads/main/tests/grades_test1.jpg?token=GHSAT0AAAAAAC53FQC5M7DTKQU4LKYGOPV2Z4XVBWA"/></td>
    <td width=40% valign="center"><img src="https://raw.githubusercontent.com/eslamwageh/Grades-auto-filler/refs/heads/main/tests/features%26classifier_result.png?token=GHSAT0AAAAAAC53FQC4YOVCYJZ5UAYIMXOAZ4XWEOQ"/></td>
  </tr>
</table>

<h4 align=center> <img  align= center width=20px height=20px src="https://media1.giphy.com/media/ZecwzuvmRrjOHsXNcI/giphy.gif?cid=6c09b9523btueuk8qe6usw2cnpb7qn8ki6evjwp62n2xiyi7&rid=giphy.gif&ct=s">Graded Sheet using OCR<a id = "results"></a>
</h4>

<table>
  <tr>
    <td width=40% valign="center"><img src="https://raw.githubusercontent.com/eslamwageh/Grades-auto-filler/refs/heads/main/tests/grades_test1.jpg?token=GHSAT0AAAAAAC53FQC4IQP7M2FORWVJOJQMZ4XWHAQ"/></td>
    <td width=40% valign="center"><img src="https://raw.githubusercontent.com/eslamwageh/Grades-auto-filler/refs/heads/main/tests/OCR_result.png?token=GHSAT0AAAAAAC53FQC4XQYDRGBP5IA7CH2YZ4XWHSA"/></td>
  </tr>
</table>

***

<h2 align=center > <img  align=center width=50px height=50px src="https://user-images.githubusercontent.com/72309546/214907431-b4e250f1-9b3a-4149-b7b4-bbd17b833b97.png">Bubble Sheet Model <a id = "bubble_sheet"></a>
</h2>

### <img  align= center width=30px height=30px src="https://media1.giphy.com/media/3ohzdWYbITsO2Y5rbi/giphy.gif?cid=6c09b9523ys3hxe1y9ueyo5ab7nzkbhv9oev797jvb5bz6rt&rid=giphy.gif&ct=s"> OverView<a id = "bubble_sheet_overview"></a>

- It handles different ink colors
- It allows different formats for the sheet ( but bubbles must be vertically aligned in all formats )
- Differnet number of questions
- Differnet number of choices
- It handles Skewing and orientation
- Printed Student ID is detected from the shaded circles

***
### <img  align= center width=50px height=50px src="https://cdn-icons-png.flaticon.com/512/1589/1589689.png"> Results<a id = "bubble_sheet_results"></a>

<h4 align=center> <img  align= center width=20px height=20px src="https://media1.giphy.com/media/ZecwzuvmRrjOHsXNcI/giphy.gif?cid=6c09b9523btueuk8qe6usw2cnpb7qn8ki6evjwp62n2xiyi7&rid=giphy.gif&ct=s"> Bubble Sheet (1) </h4>
<table>
 <thead>
    <tr>
      <th>Input</th>
      <th>Result</th>
    </tr>
   </thead>
  <tr>
    <td width=50% valign="center"><img src="https://raw.githubusercontent.com/eslamwageh/Grades-auto-filler/refs/heads/main/tests/1.jpg?token=GHSAT0AAAAAAC53FQC576IJKDECBRTFCROUZ4XWPJA"/></td>
    <td width=50% valign="center"><img src="https://github.com/eslamwageh/Grades-auto-filler/blob/main/tests/test1_result.png?raw=true"/></td>
  </tr>
</table>


<h4 align=center> <img  align= center width=20px height=20px src="https://media1.giphy.com/media/ZecwzuvmRrjOHsXNcI/giphy.gif?cid=6c09b9523btueuk8qe6usw2cnpb7qn8ki6evjwp62n2xiyi7&rid=giphy.gif&ct=s"> Bubble Sheet (2)
</h4>

<table>
 <thead>
    <tr>
      <th>Input</th>
      <th>Result</th>
    </tr>
   </thead>
  <tr>
    <td width=50% valign="center"><img src="https://github.com/eslamwageh/Grades-auto-filler/blob/main/tests/25.jpg?raw=true"/></td>
    <td width=50% valign="center"><img src="https://github.com/eslamwageh/Grades-auto-filler/blob/main/tests/test2_result.png?raw=true"/></td>
  </tr>
</table>

***

### <img  align= center width=50px src="https://i2.wp.com/www.rankred.com/wp-content/uploads/2019/07/AI-solves-Rubik-Cube.gif?fit=800%2C433&ssl=1">Demo Video<a id = "video_demo"></a>
<br>

[bubble sheet demo video.webm]()

<br>
Backup link : <a href="https://www.youtube.com/watch?v=WZZoWZTEEj0"> Demo </a>

***
 
## <img  align= center width=30px height=30px src="https://media1.giphy.com/media/WFZvB7VIXBgiz3oDXE/giphy.gif?cid=6c09b952tmewuarqtlyfot8t8i0kh6ov6vrypnwdrihlsshb&rid=giphy.gif&ct=s"> Contributors <a id = "contributors"></a>

<table>
<tr>
  <td align="center">
        <a href="https://github.com/eslamwageh">
            <img src="https://avatars.githubusercontent.com/u/53353517?v=4" width="100;" alt="EssamWisam"/>
            <br />
            <sub><b>Eslam Wageh</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Mina-H-William">
            <img src="https://avatars.githubusercontent.com/u/118685507?v=4" width="100;" alt="Kariiem"/>
            <br />
            <sub><b>Mina Hany William</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Ashraf-Bahy">
            <img src="https://avatars.githubusercontent.com/u/111181298?v=4" width="100;" alt="Muhammad-saad-2000"/>
            <br />
            <sub><b>Ashraf Bahy</b></sub>
        </a>
    </td>
    </tr>
</table>


## <img  align= center width=50px height=50px src="https://media1.giphy.com/media/ggoKD4cFbqd4nyugH2/giphy.gif?cid=6c09b9527jpi8kfxsj6eswuvb7ay2p0rgv57b7wg0jkihhhv&rid=giphy.gif&ct=s"> License <a id = "license"></a>
This software is licensed under MIT License, See [License](https://github.com/eslamwageh/Grades-auto-filler/blob/main/LICENSE) for more information ©Eslam Wageh.
