const tf = require('@tensorflow/tfjs-node');
const { Storage } = require('@google-cloud/storage');
const { v4: uuidv4 } = require('uuid');
const admin = require('firebase-admin');
const path = require('path');
const fs = require('fs');

// Inisialisasi Firebase Admin SDK untuk Firestore
admin.initializeApp({
  credential: admin.credential.applicationDefault(),
});
const db = admin.firestore();

// Cloud Storage setup
const storage = new Storage();
const bucketName = '<BUCKET_NAME>'; // Ganti dengan nama bucket Cloud Storage Anda

let model;

// Load TensorFlow.js model from Cloud Storage
const loadModel = async () => {
  const file = storage.bucket(bucketName).file('model/model.json'); // Sesuaikan dengan path model
  const [contents] = await file.download();
  model = await tf.loadGraphModel(`file://${__dirname}/models/model.json`);
  console.log('Model loaded successfully');
};

// Handler untuk prediksi
const predictHandler = async (req, res) => {
  try {
    if (!req.file) throw new Error('No file uploaded');

    // Preprocess the uploaded image
    const buffer = req.file.buffer;
    const image = tf.node.decodeImage(buffer).resizeNearestNeighbor([224, 224]).expandDims(0);

    // Make prediction
    const prediction = model.predict(image).dataSync();
    const result = prediction[0] > 0.5 ? 'Cancer' : 'Non-cancer';

    // Simpan data prediksi ke Firestore
    const id = uuidv4();
    await db.collection('predictions').doc(id).set({
      id: id,
      result: result,
      suggestion: result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.',
      createdAt: new Date().toISOString(),
    });

    // Prepare response
    const response = {
      status: 'success',
      message: 'Model is predicted successfully',
      data: {
        id: id,
        result: result,
        suggestion: result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.',
        createdAt: new Date().toISOString(),
      },
    };

    res.json(response);
  } catch (error) {
    res.status(400).json({
      status: 'fail',
      message: error.message || 'Terjadi kesalahan dalam melakukan prediksi',
    });
  }
};

// Handler untuk riwayat prediksi
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

// Handler untuk file terlalu besar
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
