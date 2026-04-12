import React, { useEffect, useState } from 'react';
import AddDocumentForm from './AddDocumentForm';
import api from '../api';

const DocumentList = () => {
  const [documents, setDocuments] = useState([]);

  const addDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      await api.post('/documents', formData);
    } catch (error) {
      console.error("Upload Fail", error);
    }
  };


  return (
    <div>
      <h2>Upload Documents to the Database:</h2>
      <ul>
        {documents.map((document, index) => (
          <li key={index}>{document.title}</li>
        ))}
      </ul>
      <AddDocumentForm addDocument={addDocument} />
    </div>
  );
};

export default DocumentList;