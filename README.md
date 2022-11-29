# Number Plate Detection_RSA

Install the following code to install python packages to run the project

pip install opencv -python

pip install pytesseract

pip install numpy

pip install matplotlib

pip install flask

pip install firebase-admin

Once you install the above packages please find the tesseract ocr folder in your Program Files Folder

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

Instead of code replace above path with your .exe file location path

# If you want to run Main.ipyn file

If you to run the .ipyn file, Please open the anaconda promot and paste the folder file path 

After that open  jupyter notebook using anaconda promote so that you do neet change any files paths in the code

Please paste the prefered image in the input_images folder and give that as input in the last cell of  main.ipynb file

# If you want to run full application

Open your command promot on your system and open the flask in your command promot

Type python appy.py in the commond promot, You will get the IP address. 

Please type the IP address in your preferred brower and continue the web app

# Details about the project

We take input from the web application, for web application we used flask

For storing login details we used firebase as a database. we store all the data in the firebase.

 <img width="551" alt="main11" src="https://user-images.githubusercontent.com/110091047/204408101-5e229e87-eacb-4586-b3ea-90634b52f723.png">
 
In this we have used yolo v5 model(you can the implementation process in Training_yolo5_model.ipynb) for obejct detection
 
If you upload the image the image will be loaded to the YOLO V5 model.

<img width="945" alt="main" src="https://user-images.githubusercontent.com/110091047/204408300-560e39bf-bee0-4685-b727-55207aa880b6.png">

And We Used pytesseract for the detected number plate from the model.

Once we extract the number palte. We send this as a input to RSA function which helps to encrypt the text 

The encrypted text will be stored in the database and once you press the decrypt in the web app. You will get the decrypted output and it will be stored in the Firebase database aswell.
<img width="905" alt="second" src="https://user-images.githubusercontent.com/110091047/204407990-64ee2d6e-50c7-42f0-8104-303fb2454797.png">

<img width="904" alt="03" src="https://user-images.githubusercontent.com/110091047/204408054-f816bb70-9db6-41d5-964a-47815919110c.png">

As we use firebase as a databse even if your try to implement in your system all the data will be stored in original database.

If you want have an access to the firebase check if the data stored in the firebase. You can see the following attachement.

<img width="926" alt="databse" src="https://user-images.githubusercontent.com/110091047/204407908-36a15845-93f4-430f-8fd1-e5a5a7c5e5d6.png">
