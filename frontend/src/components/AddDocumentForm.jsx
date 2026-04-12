import React, { useState } from 'react';

const AddDocumentForm = ({ addDocument }) => {
  const [file, setFile] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    if (file) {
      addDocument(file);
      //FileSystemFileEntry(null);
      setFile(null);
      event.target.reset();
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="file"
        accept = ".txt" //only.txt files
        onChange={(e) => setFile(e.target.files[0])}
        placeholder="Upload document name"
      />
      <button type="submit">Upload text file to Drive</button>
    </form>
  );
};

export default AddDocumentForm;