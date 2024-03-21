import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import openai
import json
import requests
from typing import Dict, List
from streamlit_option_menu import option_menu

from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools  # type: ignore
from langchain.tools import BaseTool
import secrets

def main():
    
    openai.api_key = secrets.OPENAI_API_KEY
    
    st.title("Agri Aid Expert")
    
    tab1 , tab2 , tab4 , tab5 , tab6 = st.tabs(["**Home**", "**AI Detection**", "**Crop Recommendation**", "**Expert Advice**", "**News**"])
    
    with tab1:
        st.title("Home")
        st.write("**Welcome to 'Agri Aid Expert,' your all-in-one agricultural companion. Our cutting-edge app employs AI technology to identify and combat plant diseases, offering Gen AI solutions for healthier crops. Connect with local agricultural experts, stay informed with the latest news, and make data-driven planting decisions with our ML-powered crop recommendations. Excitingly, we're planning to launch an online selling store for farmers in the future. Join us today to revolutionize your farming experience and unlock the potential of modern agriculture**")
        
    with tab2:
        st.title("AI Detection")
        selected_option = st.selectbox(
        'Select the type of disease you want to detect',
        ('Apple', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato')
        )
        uploaded_file = st.file_uploader("Capture an image...", type="jpg")
        
        result = ""
        
        if st.button("Detect"):
            if selected_option == 'Apple':
                res = detect_apple(uploaded_file)
                result = result + res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Corn':
                res = detect_corn(uploaded_file)
                result = result + res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Grape':
                res = detect_grape(uploaded_file)
                result = result + res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Cherry':
                res = detect_cherry(uploaded_file)
                result = result + res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Pepper':
                res = detect_pepper(uploaded_file)
                result = result + res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Potato':
                res = detect_potato(uploaded_file)
                result = result + res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Strawberry':
                res = detect_strawberry(uploaded_file)
                result = result + res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Tomato':
                st.write(uploaded_file)
                st.write("Tomato is healthy")
            elif selected_option == 'Peach':
                res = detect_peach(uploaded_file)
                result = result + res
                st.write(f"Predicted Class: {res}")
                
           
        if st.button("Get Solution"):
            soln = get_solution(result,selected_option)
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write(soln)
            
         
        
    with tab4:        
        st.title("Crop Recommendation")
        size = st.text_input("Enter the size of your farm")
        climate =  st.text_input("Enter the climate in your area")
        location =  st.text_input("Enter your location")
        budget = st.text_input("Enter the budget you have")
        
        if st.button("Submit"):
            res = crop_recommendation(size , climate , location , budget)
            st.write(res)
        
        
    with tab5:
        st.title("Expert Advice")
        location = st.text_input("Enter your location to get details of expert advice")
        
        if st.button("Search Expert"):
            res = search_expert(location)
            #write the result of the dictionary
            st.markdown("---")
            for i in res:
                st.write("**Title** : " + i['title'])
                st.write("**Address** : " + i['address'])
                #check if phoneNumber key available means display it
                if 'phoneNumber' in i:
                    st.write("**Phone Number** : " + i['phoneNumber'])
                else:
                    st.write("**Phone Number** : Not Available")
                if 'category' in i:
                    st.write("**Category** : " + i['category'])
                else:
                   st.write("**Category** : Not Available")
                if 'rating' in i:
                    st.write("**Rating** : " + str(int(i['rating'])))
                else:
                    st.write("**Rating** : Not Available")
                if 'website' in i:
                    st.write("**Website** : " + i['website'])
                else:
                    st.write("**Website** : Not Available")
                st.markdown("---")
            
    with tab6:
        st.title("News")
        st.write("Explore the latest news in the field of agriculture")
        search = st.text_input("Enter the topic you want to search")
        
        if st.button("Search News"):
            res = search_news(search)
            
            st.markdown("---")
            for i in res:
                st.write("**Title** : " + i['title'])
                st.write("**Link** : " + i['link'])
                st.write("**Snippet** : " + i['snippet'])
                st.markdown("---")   



#crop recommendation function

def crop_recommendation(size , climate , location , budget):
    
    query = "i have a "+size+" acres of land and i have a budget of Rs:"+budget+" in a "+climate+" season in "+location+" which crop is best suited to my field. i need answer in a single word and a brief description about the reason why that crop?"
    
    messages = [
        {"role" : "system", "content" : "You are a kind healpful assistant."},
    ]
    
    messages.append(
            {"role" : "user", "content" : query},
        )
    
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    
    reply = chat.choices[0].message.content
    
    print(reply)
    
    return reply


#expert advice details

def search_expert(location):
    
    url = "https://google.serper.dev/places"

    payload = json.dumps({
    "q": "agronomists in "+location,
    })
    headers = {
    'X-API-KEY': secrets.SERPER_API_KEY,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    
    res = json.loads(response.text)
    
    return res['places']


    
#news search in the field of agriculture

def search_news(search):
    
    url = "https://google.serper.dev/news"

    payload = json.dumps({
    "q": "agricultural experts addresses in "+search,
    })
    headers = {
    'X-API-KEY': secrets.SERPER_API_KEY,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    
    res = json.loads(response.text)
    
    return res['news']


def detect_apple(uploaded_image):
    
    if uploaded_image is not None:
        # Load your trained model
        model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\apple.h5')

        # Perform classification when the user uploads an image
        img = image.load_img(uploaded_image, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Perform the prediction
        predictions = model.predict(img_array)

        # Define the class labels
        class_labels = ["Apple scab", "Apple Black rot", "Apple rust", "Healthy apple leaf"]

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Display the predicted class label
        return predicted_class

def detect_corn(uploaded_image):
    
    if uploaded_image is not None:
        # Load your trained model
        model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\corn.h5')

        # Perform classification when the user uploads an image
        img = image.load_img(uploaded_image, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Perform the prediction
        predictions = model.predict(img_array)

        # Define the class labels
        class_labels = ["Cercospora leaf spot Gray leaf spot", "Common rust", "Northern Leaf Blight", "Healthy corn leaf"]

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Display the predicted class label
        return predicted_class
        
def detect_grape(uploaded_image):
        
        if uploaded_image is not None:
            # Load your trained model
            model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\grape.h5')
    
            # Perform classification when the user uploads an image
            img = image.load_img(uploaded_image, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values
    
            # Perform the prediction
            predictions = model.predict(img_array)
    
            # Define the class labels
            class_labels = ["Black rot", "Esca (Black Measles)", "Leaf blight (Isariopsis Leaf Spot)", "Healthy grape leaf"]
    
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)
    
            # Get the predicted class label
            predicted_class = class_labels[predicted_class_index]
    
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
            # Display the predicted class label
            return predicted_class
            
def detect_cherry(uploaded_image):
        
    
    if uploaded_image is not None:
        
        model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\cherry.h5')
        
        img = image.load_img(uploaded_image, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        predictions = model.predict(img_array)
        
        if predictions[0][0] == 1:
            predicted_class = "Cherry healthy"
        else:
            predicted_class = "Cherry Powdery Mildew disease"
            
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        return predicted_class
    
def detect_peach(uploaded_image):
        
         if uploaded_image is not None:
            # Load your trained model
            model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\peach.h5')
    
            # Perform classification when the user uploads an image
            img = image.load_img(uploaded_image, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values
    
            # Perform the prediction
            predictions = model.predict(img_array)
    
            # Define the class labels
            class_labels = ["Peach Bacterial spot", "Healthy peach leaf"]
    
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)
    
            # Get the predicted class label
            predicted_class = class_labels[predicted_class_index]
    
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
            # Display the predicted class label
            return predicted_class
        
#pepper

def detect_pepper(uploaded_image):
        
         if uploaded_image is not None:
            # Load your trained model
            model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\pepper.h5')
    
            # Perform classification when the user uploads an image
            img = image.load_img(uploaded_image, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values
    
            # Perform the prediction
            predictions = model.predict(img_array)
    
            # Define the class labels
            class_labels = ["Pepper Bacterial spot", "Healthy pepper leaf"]
    
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)
    
            # Get the predicted class label
            predicted_class = class_labels[predicted_class_index]
    
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
            # Display the predicted class label
            return predicted_class


#strawberry


def detect_strawberry(uploaded_image):
        
         if uploaded_image is not None:
            # Load your trained model
            model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\strawberry.h5')
    
            # Perform classification when the user uploads an image
            img = image.load_img(uploaded_image, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values
    
            # Perform the prediction
            predictions = model.predict(img_array)
    
            # Define the class labels
            class_labels = ["Strawberry Leaf scorch", "Healthy strawberry leaf"]
    
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)
    
            # Get the predicted class label
            predicted_class = class_labels[predicted_class_index]
    
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
            # Display the predicted class label
            return predicted_class
    
    
#potato

def detect_potato(uploaded_image):
    
    if uploaded_image is not None:
            # Load your trained model
            model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\potato.h5')
    
            # Perform classification when the user uploads an image
            img = image.load_img(uploaded_image, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values
    
            # Perform the prediction
            predictions = model.predict(img_array)
    
            # Define the class labels
            class_labels = ["Potato Early blight", "Potato Late blight", "Healthy potato leaf"]
    
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)
    
            # Get the predicted class label
            predicted_class = class_labels[predicted_class_index]
    
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
            # Display the predicted class label
            return predicted_class
    
    
def get_solution(res,plant):
    
    query = "I am a farmer , my "+plant+" plant/crop has this "+res+". Give me reply in three passage Solution should contain solution to cure the disease, Symptoms should contain symptoms and reasons why that disease occur and Prevention should contain prevention to avoid the disease that should not come again in future."
   
    messages = [
        {"role" : "system", "content" : "You are a kind healpful assistant."},
    ]
    
    messages.append(
            {"role" : "user", "content" : query},
        )
    
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    
    reply = chat.choices[0].message.content
    
    print(reply)
    
    return reply




if __name__ == "__main__":
    main()