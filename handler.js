const tf = require('@tensorflow/tfjs-node');
const { Storage } = require('@google-cloud/storage');
const { v4: uuidv4 } = require('uuid');
const admin = require('firebase-admin');
const path = require('path');
const fs = require('fs');

// Inisialisasi Firebase Admin SDK menggunakan default credentials
// admin.initializeApp({
//   credential: admin.credential.applicationDefault(),
//   projectId: process.env.GOOGLE_CLOUD_PROJECT || 'submissionmlgc-akbarpermana',
// });

// Path ke file service account
const serviceAccountPath = '/root/config/service-account-key.json'; 

admin.initializeApp({
  credential: admin.credential.cert(require(serviceAccountPath)),
});

const db = admin.firestore();

// Cloud Storage setup menggunakan default credentials
const storage = new Storage(); 
const bucketName = 'storagepenerapan-ml'; // Nama bucket Cloud Storage
const modelFolder = path.join(__dirname, 'local_model'); // Folder lokal untuk menyimpan model

let model;

// ngunduh file dari Cloud Storage
const downloadFile = async (bucketName, fileName, destination) => {
  const bucket = storage.bucket(bucketName);
  const file = bucket.file(fileName);
  await file.download({ destination });
  console.log(`File ${fileName} downloaded to ${destination}`);
};

// Fungsi ngambil model dari Cloud Storage
const loadModel = async () => {
  try {
    if (!fs.existsSync(modelFolder)) {
      fs.mkdirSync(modelFolder);
    }

    console.log('Downloading model.json...');
    const modelJsonPath = path.join(modelFolder, 'model.json');
    await downloadFile(bucketName, 'model.json', modelJsonPath);

    for (let i = 1; i <= 4; i++) {
      const shardFileName = `group1-shard${i}of4.bin`;
      const shardFilePath = path.join(modelFolder, shardFileName);
      console.log(`Downloading ${shardFileName}...`);
      await downloadFile(bucketName, shardFileName, shardFilePath);
    }

    model = await tf.loadGraphModel(`file://${modelJsonPath}`);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
};

//menangani prediksi
const predictHandler = async (req, res) => {
  try {
    if (!req.file) throw new Error('No file uploaded');

    console.log('Buffer size:', req.file.buffer.length);

    // Preprocess gambar yang diunggah
    const buffer = req.file.buffer;
    let image = tf.node.decodeImage(buffer).resizeNearestNeighbor([224, 224]).expandDims(0);
    image = image.toFloat(); // Konversi tipe data ke float32
    console.log('Image tensor shape:', image.shape);


    if (!model) {
      throw new Error('Model is not loaded. Please ensure the model is properly initialized.');
    }

    // Prediksi model
    const prediction = model.predict(image).dataSync();
    console.log('Prediction result:', prediction);

    // hasil prediksi
    const result = prediction[0] > 0.5 ? 'Cancer' : 'Non-cancer';
    console.log('Classification result:', result);

    // nyimpen prediksi ke Firestore
    const id = uuidv4();
    const firestoreData = {
      id: id,
      result: result,
      suggestion: result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.',
      createdAt: new Date().toISOString(),
    };

    await db.collection('predictions').doc(id).set(firestoreData);

  
    const response = {
      status: 'success',
      message: 'Model is predicted successfully',
      data: firestoreData,
    };

    res.json(response);
  } catch (error) {
    console.error('Prediction error:', error.message);
    res.status(400).json({
      status: 'fail',
      message: error.message || 'Terjadi kesalahan dalam melakukan prediksi',
    });
  }
};

//menanganin riwayat prediksi
const historyHandler = async (req, res) => {
  try {
    const snapshot = await db.collection('predictions').get();
    const histories = snapshot.docs.map(doc => ({
      id: doc.id,
      history: doc.data(),
    }));

    res.json({
      status: 'success',
      data: histories,
    });
  } catch (error) {
    res.status(400).json({
      status: 'fail',
      message: error.message || 'Terjadi kesalahan saat mengambil riwayat',
    });
  }
};

// nanganin error file terlalu besar
const errorHandler = (err, req, res, next) => {
  if (err.code === 'LIMIT_FILE_SIZE') {
    res.status(413).json({
      status: 'fail',
      message: 'Payload content length greater than maximum allowed: 1000000',
    });
  } else {
    next(err);
  }
};

module.exports = {
  loadModel,
  predictHandler,
  historyHandler,
  errorHandler,
};
