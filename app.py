from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
import numpy as np
from PIL import Image
import base64
import io
import tensorflow as tf
from tensorflow import keras
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

class ImagePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.img_size = None
        self.loaded = False
        
    def load_model(self):
        try:
            logger.info("🔄 Cargando modelo...")
            
            required_files = [
                'simplified_classifier_model.h5',
                'simplified_label_encoder.pkl',
                'simplified_config.pkl'
            ]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                logger.error(f"❌ Archivos no encontrados: {missing_files}")
                logger.error("🔧 Ejecuta el entrenamiento primero para generar estos archivos")
                return False
            
            self.model = keras.models.load_model('simplified_classifier_model.h5')
            logger.info("✅ Modelo cargado")
            
            with open('simplified_label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info("✅ Label encoder cargado")
            
            with open('simplified_config.pkl', 'rb') as f:
                config = pickle.load(f)
                self.img_size = config['img_size']
            logger.info("✅ Configuración cargada")
            
            self.loaded = True
            logger.info(f"🎯 Clases disponibles: {list(self.label_encoder.classes_)}")
            logger.info("🚀 Sistema listo para hacer predicciones!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            logger.error(f"🔍 Detalles: {traceback.format_exc()}")
            return False
    
    def predict_image(self, image_base64: str) -> dict:
        if not self.loaded:
            return {
                "success": False,
                "error": "Modelo no cargado. El sistema no está listo."
            }
        
        try:
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            try:
                img_data = base64.b64decode(image_base64)
            except Exception as e:
                return {
                    "success": False,
                    "error": "Imagen base64 inválida"
                }
            
            img = Image.open(io.BytesIO(img_data))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize(self.img_size, Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            results = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                percentage = float(predictions[i] * 100)
                results[class_name.lower()] = round(percentage, 1)
            
            sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
            
            formatted_parts = []
            for class_name, percentage in sorted_results.items():
                formatted_parts.append(f"{percentage}% {class_name}")
            
            formatted_result = ", ".join(formatted_parts)
            
            logger.info(f"🎯 Predicción realizada: {formatted_result}")
            
            return {
                "success": True,
                "predictions": sorted_results,
                "top_prediction": {
                    "class": list(sorted_results.keys())[0],
                    "confidence": list(sorted_results.values())[0]
                },
                "formatted_result": formatted_result
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando imagen: {e}")
            return {
                "success": False,
                "error": f"Error procesando imagen: {str(e)}"
            }

predictor = ImagePredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not predictor.loaded:
            return jsonify({
                "success": False,
                "error": "Modelo no cargado. Reinicia el servidor."
            }), 500
        
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type debe ser application/json"
            }), 400
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "Campo 'image' requerido en el JSON"
            }), 400
        
        image_base64 = data['image']
        
        if not image_base64 or not isinstance(image_base64, str):
            return jsonify({
                "success": False,
                "error": "El campo 'image' debe ser un string base64 válido"
            }), 400
        
        result = predictor.predict_image(image_base64)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "predictions": result["predictions"],
                "top_class": result["top_prediction"]["class"],
                "confidence": result["top_prediction"]["confidence"]
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Error en endpoint /predict: {e}")
        logger.error(f"🔍 Detalles: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": "Error interno del servidor"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint no encontrado",
        "available_endpoints": ["/", "/predict", "/health"]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Método no permitido",
        "hint": "El endpoint /predict solo acepta POST"
    }), 405

def main():
    if predictor.load_model():
        app.run(
            host='0.0.0.0',
            port=4000,
            debug=False,
            threaded=True
        )
    else:
        print("❌ No se pudo cargar el modelo!")
        print("🔧 Soluciones:")
        print("   1. Ejecuta el entrenamiento: python simplified_classifier.py")
        print("   2. Verifica que existen los archivos:")
        print("      - simplified_classifier_model.h5")
        print("      - simplified_label_encoder.pkl")
        print("      - simplified_config.pkl")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Servidor detenido por el usuario")
    except Exception as e:
        print(f"\n💥 Error iniciando servidor: {e}")
        logger.error(f"Error fatal: {traceback.format_exc()}")