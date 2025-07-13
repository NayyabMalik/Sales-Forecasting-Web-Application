// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize toast
    var toastEl = document.getElementById('liveToast');
    var toast = new bootstrap.Toast(toastEl, {
        autohide: true,
        delay: 3000
    });
    
    // Store toast instance for global access
    window.appToast = toast;
});

// Save graph function
function saveGraph(url, filename) {
    // Extract the actual image path from the URL
    const imagePath = url.split('/static/')[1];
    const fullUrl = window.location.origin + '/static/' + imagePath;
    
    // Create a temporary link to download the image
    const link = document.createElement('a');
    link.href = fullUrl;
    link.download = filename.replace(/[^a-z0-9]/gi, '_').toLowerCase() + '.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Show toast notification
    if (window.appToast) {
        window.appToast.show();
    }
}

// Make graphs clickable for fullscreen view
document.querySelectorAll('.viz-container img').forEach(img => {
    img.addEventListener('click', function() {
        const modal = document.createElement('div');
        modal.style.position = 'fixed';
        modal.style.top = '0';
        modal.style.left = '0';
        modal.style.width = '100%';
        modal.style.height = '100%';
        modal.style.backgroundColor = 'rgba(0,0,0,0.9)';
        modal.style.display = 'flex';
        modal.style.flexDirection = 'column';
        modal.style.justifyContent = 'center';
        modal.style.alignItems = 'center';
        modal.style.zIndex = '2000';
        modal.style.cursor = 'zoom-out';
        
        const modalImg = document.createElement('img');
        modalImg.src = this.src;
        modalImg.style.maxWidth = '90%';
        modalImg.style.maxHeight = '90%';
        modalImg.style.objectFit = 'contain';
        
        const modalTitle = document.createElement('div');
        modalTitle.textContent = this.alt;
        modalTitle.style.color = 'white';
        modalTitle.style.marginTop = '15px';
        modalTitle.style.fontSize = '1.2rem';
        modalTitle.style.textAlign = 'center';
        
        modal.appendChild(modalImg);
        modal.appendChild(modalTitle);
        modal.addEventListener('click', function() {
            document.body.removeChild(modal);
        });
        
        document.body.appendChild(modal);
    });
});
