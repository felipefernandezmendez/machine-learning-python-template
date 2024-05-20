import streamlit as st
from PIL import Image
import requests
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
from utils import load_image, load_ben_color
import gdown

#model = load_model("path_to_your_model.h5")  # Cambia a la ruta de tu modelo
url= "https://drive.google.com/file/d/1ELGWX058ElB9eAe-MbfMuIUgUAEVdVP4/view?usp=sharing"
output= "Modelo_DesNet121"
gdown.download(url, output, quiet=False)
#model = load_model(output)


def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation URL
lottie_coding = load_lottie_url("https://lottie.host/132a64cd-b421-46f4-99ec-8387c1ced1af/4L0M7FMGWD.json")


def main():
    st.title("Detección de Retinopatía Diabética")
    st_lottie(lottie_coding, height=300, key='coding')
    
    
    st.write("---")
        #left_column, right_column = st.columns(2)
        #with left_column:
    
    st.write("Sube una imagen para que el modelo haga una predicción.")
    st.header("Algún texto que quede bien aquí")
    st.write("Aquí se puede poner un texto")
    # Carga de imagen
    
    image_file = st.file_uploader("Carga una imagen", type=["jpg", "jpeg", "png"])
   
    if image_file is not None:
        # Mostrar la imagen cargada
        img = Image.open(image_file)
        st.image(img, caption="Imagen cargada", use_column_width=True)
    # Preprocesar la imagen
        img_array = np.array(img)
        processed_image =load_ben_color(img_array)
        processed_image = np.expand_dims(processed_image, axis=0)
        # Mostrar la imagen preprocesada
        st.write("Solo para prueba, aquí está la imagen preprocesada.")
        st.image(processed_image[0], caption="Imagen preprocesada", use_column_width=True)

    
    #with st.container():
    st.write("---")
    


        

if __name__ == "__main__":
    main()

