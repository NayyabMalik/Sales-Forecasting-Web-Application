        // Handle file drop area highlighting
        const fileDropArea = document.getElementById('fileDropArea');
        const fileInput = document.getElementById('file');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileDropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            fileDropArea.classList.add('bg-light');
        }
        
        function unhighlight() {
            fileDropArea.classList.remove('bg-light');
        }
        
        // Handle dropped files
        fileDropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            fileInput.files = files;
            updateFilePreview(files);
        }
        
        // Handle selected files
        fileInput.addEventListener('change', function() {
            updateFilePreview(this.files);
        });
        
        function updateFilePreview(files) {
            const filePreview = document.getElementById('filePreview');
            filePreview.innerHTML = '';
            
            if (files.length > 0) {
                const fileList = document.createElement('div');
                fileList.className = 'list-group';
                
                for (let i = 0; i < files.length; i++) {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'list-group-item';
                    fileItem.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <i class="fas fa-file-alt me-2"></i>
                                ${files[i].name}
                            </div>
                            <span class="badge bg-secondary">${formatFileSize(files[i].size)}</span>
                        </div>
                    `;
                    fileList.appendChild(fileItem);
                }
                
                filePreview.appendChild(fileList);
            }
        }
        
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }