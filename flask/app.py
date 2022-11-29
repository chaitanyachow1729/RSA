from flask import Flask, render_template, request, jsonify, flash, redirect, session
import os 
from deeplearning import object_detection,rsa_implementation
from firebase_admin import credentials, firestore, initialize_app

""" Initializing the APP """
app = Flask(__name__)
app.secret_key = os.urandom(16)
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

""" Connecting to Firebase"""
cred = credentials.Certificate('key2.json')

""" Initializing the Firebase APP """
default_app = initialize_app(cred)
db = firestore.client()

users = db.collection('users')

""" Route for Index Page - Signin/Signup """
@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        data = request.form.to_dict(flat=True)
        users_ref = db.collection(u'users')
        docs = users_ref.stream()
        try:
            for doc in docs:
                if doc.id==data["username"]:
                    user_data = doc.to_dict()
                    print(user_data)
                    if user_data["password"]==data["password"]:
                        session['loggedin'] = True
                        session['username'] = data["username"]
                        return redirect("/home")
                    else:
                        raise Exception("Invalid Password")
            raise Exception("Username not Found, Please Sign Up!")
        except Exception as e:
            flash(str(e), 'error')
        return redirect("/")
    return render_template('index.html')

""" Route for Signup Page """
@app.route('/signup',methods=['POST','GET'])
def signup():
    if request.method == 'POST':
        users_ref = db.collection(u'users')
        docs = users_ref.stream()
        data = request.form.to_dict(flat=True)
        try:
            username = data['username']
            for doc in docs:
                if doc.id==username:
                    raise Exception("User with Id %s Already Exists"%username)
            
            users.document(username).set(data)
            flash("User account created Please Login!", 'success')
            return redirect("/")
        except Exception as e:
            flash(e)
    
    return render_template('signup.html')

""" Route for Logging out User """
@app.route('/logout')
def logout():
    session['loggedin'] = None
    return redirect("/")

""" Route for Home Page """
@app.route('/home',methods=['POST','GET'])
def home():
    if session.get("loggedin") is None:
        return redirect("/")
    session['encrypted'] = None
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        text_list = object_detection(path_save,filename)
        
        my_string = " ".join(text_list)

        crypted_msg,decrypted = rsa_implementation(my_string)

        #crypted_msg = rsa_implementation(text_list)
        users_ref = db.collection(u'users')
        docs = users_ref.stream()
        username = session.get("username")
        try:
            for doc in docs:
                if doc.id==username:
                    user_data = doc.to_dict()
                    user_data['encrypted_msg'] = "$$$".join(str(i) for i in crypted_msg)
                    user_data['decrypted_msg'] = decrypted
                    users.document(username).set(user_data)
                    print(user_data)
        except Exception as e:
            print(str(e))

        session["my_string"] = my_string        
        print(f"{my_string=}")

        return render_template('home.html',upload=True,upload_image=filename,text=my_string,crypted_msg=crypted_msg,decrypted=decrypted,no=len(my_string))
    
    return render_template('home.html',upload=False)

""" Route for Decrypting Text """
@app.route('/decrypt',methods=['GET'])
def decrypt():
    data=dict()
    if session.get("encrypted") is None:
        data["error"] = "Please Encrypt First!"
    elif session.get("my_string") is None:
        data["error"] = "Please Upload Image First!"
    else:
        data["success"]="Decrypted Successfully!"
    return data

""" Route for encrypting Text """
@app.route('/encrypt',methods=['GET'])
def encrypt():
    data=dict()
    if session.get("my_string") is None:
        data["error"] = "Please Upload Image First!"
        session["encrypted"] = None
    else:
        data["success"]="Encrypted Successfully!"
        session["encrypted"] = True
    return data
    


if __name__ =="__main__":
    app.run(debug=True)