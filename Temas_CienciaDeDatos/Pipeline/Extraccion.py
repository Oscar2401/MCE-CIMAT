import warnings
warnings.filterwarnings('ignore')
import torch
import tensorflow as tf
import numpy as np
import cv2
from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
import pytesseract
from PIL import Image
import os
import shutil

# ============ MODELOS ============ #
# Modelo: Yolov5
modelo_yolo = torch.hub.load('Extracion_texto_doc/YoloModel/yolov5/',
								'custom',
								path='Extracion_texto_doc/YoloModel/mi_modelo/weights/best.pt', 
								source='local') 
# Modelo: DeepErase
modelo_DE = tf.keras.models.load_model('Extracion_texto_doc/DeepEraseModel/my_model')
# Modelos: CRAFT
refine_net = load_refinenet_model()
craft_net = load_craftnet_model()

# ============ FUNCION ============ #
def extraccion_texto(imagen_path):
	"""
	Función para extraer texto de un documento.
	
	Arg
		imagen_path: Dirección al  documento.
	"""
	# Etiquetamos de documento usando modelo de Yolov5
	documento_marcado = modelo_yolo(imagen_path, augment=True)
	
	# Recortamos
	recortes = [imgCrop['im'] 
						for imgCrop in documento_marcado.crop(save=False) 
						if imgCrop['conf'] > 0.7 ]
	etiquetas = [imgCrop['label'][:len(imgCrop['label'])-5] 
						for imgCrop in documento_marcado.crop(save=False) 
						if imgCrop['conf'] > 0.7 ]
	
	
	recortes_bi = []
	for rec in recortes:
		bw = cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)
		recortes_bi.append(255-bw)
	
	Extrac_palab_x_reg = {}
	for idRec, etiqueta_nombre in enumerate(etiquetas):
	
		# ===== División de cada palabra del texto ===== #
		if etiqueta_nombre == 'NoExp':
			prediction_result = get_prediction(
										image=recortes_bi[idRec],
										craft_net=craft_net,
										refine_net=refine_net,
										text_threshold=0.1,
										link_threshold=0.4,
										low_text=0.01,
										cuda=False,
										long_size=1000)
			
		else:
			prediction_result = get_prediction(
										image=recortes_bi[idRec],
										craft_net=craft_net,
										refine_net=refine_net,
										text_threshold=0.2,
										link_threshold=0.9,
										low_text=0.2,
										cuda=False,
										long_size=5500)
		
		recortes_dir = 'output/image_crops/'
		shutil.rmtree(recortes_dir)
		#!rm -r $recortes_dir
		palabras = export_detected_regions(
									image=recortes_bi[idRec],
									regions=prediction_result["boxes"])
		
		# ===== Binarizamos las palabras ===== #
		for img in palabras:
			imagen = 255-cv2.imread(img,0)
			imagen = np.where(imagen>(np.median(imagen)-0.9*np.std(imagen)), 255, 0)
			cv2.imwrite(img,imagen)
			
		# ===== Refinamos la extracción de palabras ===== #
		# ===== y ordenamos el texto					  ===== #
		puntos = prediction_result['boxes'][:,0]
		palabras_inc = palabras.copy()
		p=0
		for i, ImPaRePath in enumerate(palabras):
			imagen = cv2.imread(ImPaRePath,0)
			if imagen.shape[1]>256 or imagen.shape[0]>85:
        
				if imagen.shape[0] <= 85:
					LongSize = 500
				else:
					LongSize = 2000
						  
				imagen[imagen==0]=10
				sub_prediction_result = get_prediction(
													image=imagen,
													craft_net=craft_net,
													refine_net=refine_net,
													text_threshold=0.2,
													link_threshold=0.9,
													low_text=0.2,
													cuda=False,
													long_size= LongSize)
				  
				recortes_dir2 = 'output/image_crops2/'
				shutil.rmtree(recortes_dir2)
				#!rm -r $recortes_dir2

				palabras2 = export_detected_regions(
									image = imagen,
									regions = sub_prediction_result["boxes"],
									output_dir = recortes_dir2)
				  
				for img in palabras2:
					imagen2 = cv2.imread(img,0)
					imagen2[imagen2==0]=255
					imagen2[imagen2==10]=0
					cv2.imwrite(img,imagen2)
					
				if len(palabras2) != 0:  
					cc = ImPaRePath.replace('output/image_crops/','').replace('.png','')+'_'
					incremento  = puntos[i+p]
					puntos = np.delete(puntos,i+p,axis=0)
					palabras_inc.pop(i+p)
					for k, img in enumerate(palabras2):
						imagen = cv2.imread(img,0)
						aa = img.replace('crop_',cc)
						cv2.imwrite(aa,imagen)
						os.remove(img)
						#!rm $img
						ll = 'output/image_crops/'
						shutil.move(aa, ll)
						#!mv $aa $ll
						puntos = np.insert(puntos, i+p+k, sub_prediction_result["boxes"][:,0][k]+incremento, axis=0)
						palabras_inc.insert(i+p+k,ll+aa.replace('output/image_crops2/image_crops/',''))
					  
					p+=k
					os.remove(ImPaRePath)
					#!rm $ImPaRePath
		
		palabras_inc = [palabras_inc[i] for i in puntos[:,0].argsort()]
		puntos = puntos[puntos[:,0].argsort()]
		dimens = (np.min(prediction_result['boxes'][:,0,1]),np.max(prediction_result['boxes'][:,3,1]))
		row_amp = np.median(prediction_result['boxes'][:,2,1]-prediction_result['boxes'][:,0,1])
		row_amp_std = np.std(prediction_result['boxes'][:,2,1]-prediction_result['boxes'][:,0,1])
		
		# Rango de lineas para lectura
		if etiquetas[idRec] == 'NoExp':
			rangos = np.arange(dimens[0],dimens[1]-dimens[0],row_amp+row_amp_std)
		elif etiquetas[idRec] == 'Ingreso':
			rangos = np.arange(dimens[0],dimens[1]-dimens[0],row_amp+row_amp_std)
		elif etiquetas[idRec] == 'Autopsia':
			rangos = np.arange(dimens[0],dimens[1]-dimens[0],row_amp+row_amp_std)
		else:
			rangos = np.arange(dimens[0],dimens[1]-dimens[0],row_amp+0.5*row_amp_std)
			
		# Ordenamiento final
		orden = []
		for i in range(len(rangos)-1):
			pintos_in = []
			row_id = []
			for j, p in enumerate(puntos[:,1]):
				if rangos[i]<= p and p < rangos[i+1]:
					pintos_in.append(puntos[:,0][j])
					row_id.append(j)
			pintos_in = np.argsort(np.array(pintos_in))
			orden = orden +  [row_id[k] for k in pintos_in]
		orden = np.array(orden)
		
		palab_orde = [palabras_inc[i] for i in orden]
		
		
		# ===================== Extraccion de palabras de cada región ===================== #
		texto_extraccion = []
		for ImPaRePath_2 in palab_orde:
			imagen = cv2.imread(ImPaRePath_2,0)
			 
			imgPalRec = tf.io.read_file(ImPaRePath_2)
			imgPalRec = tf.image.decode_jpeg(imgPalRec, channels=1)
			imgPalRec = tf.image.resize(imgPalRec, (64,256))
			imgPalRec = tf.image.convert_image_dtype(imgPalRec, tf.float32) / 255.
			imgPalRec = tf.data.Dataset.from_tensors(imgPalRec[None,:,:,:])
			imgPalRecPre = modelo_DE.predict(imgPalRec)
			img_pred =  cv2.resize(imgPalRecPre[0][:,:,0],
											(imagen.shape[1],imagen.shape[0]),
											interpolation = cv2.INTER_AREA)
			 
			img_pred[img_pred<=(np.median(img_pred)-np.std(img_pred))] = 0
			img_pred[img_pred >(np.median(img_pred)-np.std(img_pred))] = 255
			img_pred = np.array(img_pred, dtype=np.uint8)
			 
			 	 
			if etiquetas[idRec] == 'NoExp':
				custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=Exp/.NumProgresivo0123456789'
				txt1 = pytesseract.image_to_string(imagen, lang='spa', config=custom_config)
				txt2 = pytesseract.image_to_string(img_pred, lang='spa', config=custom_config)
				if not(txt1[:-2]=='' and txt2[:-2]==''):
					texto_extraccion.append((txt1[:-2],txt2[:-2]))
				
			else:
				custom_config = r'--oem 3 --psm 6'
				txt1 = pytesseract.image_to_string(imagen, lang='spa', config=custom_config)
				txt2 = pytesseract.image_to_string(img_pred, lang='spa', config=custom_config)
				if not(txt1[:-2]=='' and txt2[:-2]==''):
					texto_extraccion.append((txt1[:-2],txt2[:-2]))
		
		Extrac_palab_x_reg[etiqueta_nombre] = texto_extraccion
		
		texto_extraccion = []
		orden = []
	
	return Extrac_palab_x_reg
		















