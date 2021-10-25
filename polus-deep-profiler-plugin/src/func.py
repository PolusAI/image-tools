class deepprofiler:

    """Create output z-Normalize Images.
        Parameters
        ----------
        image : ndarray
        Path of an input image
        Returns
        -------
        normalized_img : z normalized
        Output image
        Notes
        -------
        Clipping allow to set min value and maximum intensity values
        1) clipped values [-1, 1] to [0 , 1]
       
        """

    def __init__(self, x, inputdir, maskdir, desired_size):
        self.inputdir = inputdir
        self.maskdir =  maskdir
        self.x = x
        self.imagepath = os.path.join(self.inputdir, self.x['intensity_image'])
        self.maskpath = os.path.join(self.maskdir, self.x['mask_image'])
        self.desired_size = desired_size
             
    def loading_images(self):
        br_img = BioReader(self.imagepath)
        br_img = br_img.read().squeeze()
        br_mask = BioReader(self.maskpath)
        br_mask = br_mask.read().squeeze()
        return br_img, br_mask
    
    def z_normalization(self): 
        img, mask = self.loading_images()
        znormalized = (img - np.mean(img)) / np.std(img) 
        return znormalized, mask
    
    def masking_roi(self): 
        image, mask = self.z_normalization()
        timage, tmask = image.copy(), mask.copy()
        tmask[mask !=self.x[2]] = 0
        timage[tmask!=self.x[2]] =0
        msk_img = timage[self.x[3]:self.x[3]+self.x[5], self.x[4]:self.x[4]+self.x[6]]
        tsk_img = tmask[self.x[3]:self.x[3]+self.x[5], self.x[4]:self.x[4]+self.x[6]]
        return msk_img, tsk_img
    
    
        
        
    def resizing(self):
        x, _ = self.masking_roi() 
        Y, X = x.shape
        aspectratio = Y/X
        if Y and X > self.desired_size and aspectratio < 1:
            Y = int(aspectratio * desired_size)
            X = self.desired_size
            return cv2.resize(x, dsize=(X, Y), interpolation=cv2.INTER_CUBIC)

        elif X and Y > self.desired_size and aspectratio > 1:
            X = int(X/Y * desired_size)
            Y = self.desired_size
            return cv2.resize(x, dsize=(X, Y), interpolation=cv2.INTER_CUBIC)

        elif Y > self.desired_size:
            X = int(X/Y * self.desired_size)
            Y = self.desired_size
            return cv2.resize(x, dsize=(X, Y), interpolation=cv2.INTER_CUBIC)

        elif X > self.desired_size:
            Y = int(aspectratio * X)
            X = self.desired_size
            return cv2.resize(x, dsize=(X, Y), interpolation=cv2.INTER_CUBIC)

        else:
            return x

    def zero_padding(self):
        x = self.resizing() 
        Y, X = x.shape
        ch_w = self.desired_size - X
        ch_h = self.desired_size - Y 
        top, bottom = ch_h//2, ch_h-(ch_h//2)
        left, right = ch_w//2, ch_w-(ch_w//2)
        pad_img = cv2.copyMakeBorder(x, top, bottom, left, right, 
                                     cv2.BORDER_CONSTANT,value=[0, 0, 0])          
        return pad_img




def chunker(df, batchsize):
    for pos in range(0, len(df), batchsize):
        yield df.iloc[pos:pos + batchsize] 
        
def get_model(model):
    modelname = getattr(tf.keras.applications, model)
    return modelname(weights='imagenet', include_top=False, pooling='avg')


def model_prediction(model, image):  
    features = model.predict(image)
    return features

def feature_extraction(labels, x, features):  
    imagename = x['intensity_image']
    maskname = x['mask_image']
    cellid = labels
    df = pd.DataFrame(features)
    df.insert(0, 'ImageName', imagename)
    df.insert(1, 'MaskName', maskname) 
    df.insert(2, 'Cell_ID', cellid ) 
    return df


modelname = get_model('VGG16')
IMG_SIZE = 128        
batch_size = 2
pf = chunker(df, 2)



total_feat = []
for batch in pf:
    
    roi_images =[]
    roi_labels =[]

    for i, row in batch.iterrows():

        dclass = deepprofiler(row, inputdir, maskdir, 128)

        imgpad = dclass.zero_padding()
        img = np.dstack((imgpad, imgpad))
        img = np.dstack((img, imgpad))
        
        roi_labels.append(row['label'])
        
        roi_images.append(img)

    batch_images = np.asarray(roi_images)
    batch_labels = roi_labels
    feat = model_prediction(modelname,batch_images)
    pdm = feature_extraction(batch_labels, row,  feat) 
    total_feat.append(pdm)





# Old



    
#     def resizing(self):
#         x, _ = self.masking_roi()  
#         Y, X = x.shape
#         if Y > self.desired_size:
#             aspectratio = Y/X
#             Y = self.desired_size
#             X = int( Y/aspectratio)
#             return cv2.resize(x, dsize=(X, desired_size), interpolation=cv2.INTER_CUBIC)

#         elif X > self.desired_size:
#             aspectratio = X/Y
#             X = self.desired_size
#             Y = int(X/aspectratio)
#             return cv2.resize(x, dsize=(desired_size, Y), interpolation=cv2.INTER_CUBIC)

#         elif X and Y > self.desired_size:
#             return cv2.resize(x, dsize=(desired_size, desired_size), interpolation=cv2.INTER_CUBIC)

#         else:
#             return x


    

