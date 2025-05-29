import os
from dotenv import load_dotenv
import base64
import io
import numpy as np
import mysql.connector
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import logging
from collections import Counter

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedImageClassifier:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '3306')),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'railway')
        }
        self.model = None
        self.label_encoder = None
        self.img_size = (128, 128)  # Imagen más pequeña para rapidez
        self.history = None
        
    def connect_database(self):
        try:
            connection = mysql.connector.connect(**self.db_config)
            logger.info("✅ Conexión a base de datos exitosa")
            return connection
        except mysql.connector.Error as err:
            logger.error(f"❌ Error conectando a la base de datos: {err}")
            return None
    
    def analyze_dataset(self) -> bool:
        """Analiza la distribución de clases en el dataset"""
        connection = self.connect_database()
        if not connection:
            return False
        
        cursor = connection.cursor()
        query = "SELECT img_tag FROM NeuralNetwork"
        
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            
            labels = [result[0].strip().upper() for result in results]
            class_counts = Counter(labels)
            
            logger.info("📊 ANÁLISIS DEL DATASET:")
            logger.info(f"   - Total de imágenes: {len(results)}")
            logger.info(f"   - Número de clases: {len(class_counts)}")
            
            logger.info("📈 Distribución de clases:")
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / len(results)) * 100
                logger.info(f"   - {class_name}: {count} imágenes ({percentage:.1f}%)")
            
            # Verificar que hay suficientes datos
            min_samples_per_class = min(class_counts.values())
            if min_samples_per_class < 5:
                logger.warning("⚠️ Algunas clases tienen muy pocas muestras!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en análisis: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    
    def extract_data_from_db(self):
        """Extrae y procesa datos de la base de datos"""
        connection = self.connect_database()
        if not connection:
            return [], []
        
        cursor = connection.cursor()
        query = "SELECT img, img_tag FROM NeuralNetwork"
        
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            logger.info(f"📊 Registros encontrados: {len(results)}")
            
            images = []
            labels = []
            errors = 0
            
            for idx, (img_b64, img_tag) in enumerate(results):
                try:
                    # Procesar imagen
                    if img_b64.startswith('data:image'):
                        img_b64 = img_b64.split(',')[1]
                    
                    img_data = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_data))
                    
                    if img.size[0] < 50 or img.size[1] < 50:
                        continue
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    
                    images.append(img_array)
                    labels.append(img_tag.strip().upper())
                    
                    if (idx + 1) % 100 == 0:
                        logger.info(f"📈 Procesadas {idx + 1} imágenes...")
                    
                except Exception as e:
                    errors += 1
                    logger.warning(f"⚠️ Error procesando imagen {idx + 1}: {e}")
                    continue
            
            logger.info(f"✅ Procesamiento completado:")
            logger.info(f"   - Imágenes válidas: {len(images)}")
            logger.info(f"   - Errores: {errors}")
            
            return np.array(images), labels
            
        except mysql.connector.Error as err:
            logger.error(f"❌ Error ejecutando query: {err}")
            return [], []
        finally:
            cursor.close()
            connection.close()
    
    def create_simple_model(self, num_classes: int) -> keras.Model:
        """Crea un modelo multiclase simplificado"""
        model = keras.Sequential([
            # Bloque 1 - Básico
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.1),
            
            # Bloque 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Bloque 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Clasificador
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')  # Multiclase
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs: int = 40, validation_split: float = 0.2, test_size: float = 0.15):
        logger.info("🚀 Iniciando entrenamiento del modelo simplificado...")
        
        # Análisis del dataset
        if not self.analyze_dataset():
            logger.error("❌ No se puede proceder con el entrenamiento")
            return False
        
        logger.info("📥 Extrayendo datos...")
        X, labels = self.extract_data_from_db()
        
        if len(X) < 20:
            logger.error("❌ Necesitas al menos 20 imágenes totales")
            return False
        
        # Preparar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(labels)
        y = keras.utils.to_categorical(y_encoded)
        
        num_classes = len(self.label_encoder.classes_)
        logger.info(f"🎯 Clases encontradas: {list(self.label_encoder.classes_)}")
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"📊 División de datos:")
        logger.info(f"   - Entrenamiento: {len(X_train)} imágenes")
        logger.info(f"   - Validación: {int(len(X_train) * validation_split)} imágenes")
        logger.info(f"   - Prueba: {len(X_test)} imágenes")
        
        # Crear modelo
        self.model = self.create_simple_model(num_classes)
        logger.info("🏗️ Modelo simplificado creado")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Entrenamiento
        batch_size = min(32, max(8, len(X_train) // 10))
        logger.info(f"🎯 Iniciando entrenamiento con {epochs} épocas...")
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluación
        logger.info("📊 Evaluando modelo...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"✅ Resultados finales:")
        logger.info(f"   - Pérdida: {test_loss:.4f}")
        logger.info(f"   - Precisión: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Guardar modelo
        self.save_model()
        
        return True
    
    def predict_image_base64(self, image_b64: str) -> dict:
        """
        Predice las probabilidades de todas las clases para una imagen en base64
        
        Args:
            image_b64: Imagen en formato base64
            
        Returns:
            dict: {"predictions": {"water": 86.2, "fire": 10.1, "tree": 2.9, "mountain": 0.8}}
        """
        if not self.model or not self.label_encoder:
            return {"error": "Modelo no cargado. Entrena el modelo primero."}
        
        try:
            # Procesar imagen base64
            if image_b64.startswith('data:image'):
                image_b64 = image_b64.split(',')[1]
            
            img_data = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_data))
            
            # Procesar imagen
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize(self.img_size, Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
            
            # Predicción
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Formatear resultados como porcentajes
            results = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                percentage = float(predictions[i] * 100)
                results[class_name.lower()] = round(percentage, 1)
            
            # Ordenar por probabilidad (mayor a menor)
            sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
            
            return {
                "success": True,
                "predictions": sorted_results,
                "top_prediction": {
                    "class": list(sorted_results.keys())[0],
                    "confidence": list(sorted_results.values())[0]
                }
            }
            
        except Exception as e:
            return {"error": f"Error procesando imagen: {str(e)}"}
    
    def save_model(self):
        """Guarda el modelo y el label encoder"""
        try:
            if self.model:
                self.model.save('simplified_classifier_model.h5')
                logger.info("💾 Modelo guardado como 'simplified_classifier_model.h5'")
            
            if self.label_encoder:
                with open('simplified_label_encoder.pkl', 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                logger.info("💾 Label encoder guardado como 'simplified_label_encoder.pkl'")
                
            # Guardar configuración
            config = {'img_size': self.img_size}
            with open('simplified_config.pkl', 'wb') as f:
                pickle.dump(config, f)
            logger.info("💾 Configuración guardada como 'simplified_config.pkl'")
                
            logger.info("🎉 Entrenamiento completado exitosamente!")
            
        except Exception as e:
            logger.error(f"❌ Error guardando archivos: {e}")
    
    @classmethod
    def load_model(cls):
        """Carga un modelo previamente entrenado"""
        try:
            classifier = cls()
            
            # Cargar modelo
            classifier.model = keras.models.load_model('simplified_classifier_model.h5')
            
            # Cargar label encoder
            with open('simplified_label_encoder.pkl', 'rb') as f:
                classifier.label_encoder = pickle.load(f)
            
            # Cargar configuración
            with open('simplified_config.pkl', 'rb') as f:
                config = pickle.load(f)
                classifier.img_size = config['img_size']
            
            logger.info("✅ Modelo cargado exitosamente")
            return classifier
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            return None

def main():
    print("🧠 Clasificador de Imágenes Simplificado para App")
    print("=" * 55)
    
    classifier = SimplifiedImageClassifier()
    
    # Configuración
    epochs = int(input("📊 Número de épocas (recomendado: 30-50): ") or "40")
    
    print(f"\n🚀 Entrenando modelo simplificado con {epochs} épocas...")
    print("⏱️ Modelo optimizado para uso en app...\n")
    
    success = classifier.train_model(epochs=epochs)
    
    if success:
        print("\n" + "=" * 55)
        print("🎉 ¡ENTRENAMIENTO COMPLETADO!")
        print("✅ Modelo listo para usar en app.py")
        print("📁 Archivos generados:")
        print("   - simplified_classifier_model.h5")
        print("   - simplified_label_encoder.pkl")
        print("   - simplified_config.pkl")
        print("\n🔮 Ejemplo de uso en app.py:")
        print("""
# En tu app.py:
from simplified_classifier import SimplifiedImageClassifier

# Cargar modelo entrenado
classifier = SimplifiedImageClassifier.load_model()

# Predecir imagen
result = classifier.predict_image_base64(image_base64)
print(result['predictions'])
# Output: {'water': 86.2, 'fire': 10.1, 'tree': 2.9, 'mountain': 0.8}
        """)
        print("=" * 55)
    else:
        print("\n❌ ERROR EN EL ENTRENAMIENTO")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Entrenamiento cancelado")
    except Exception as e:
        print(f"\n💥 Error: {e}")