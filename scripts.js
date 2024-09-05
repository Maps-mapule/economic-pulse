
document.getElementById('signup-form')?.addEventListener('submit', function(event) {
    var password = document.getElementById('password').value;
    var confirmPassword = document.getElementById('confirm-password').value;
    var errorMessage = document.getElementById('error-message');
    
    if (password !== confirmPassword) {
        errorMessage.textContent = 'Passwords do not match.';
        event.preventDefault(); 
    } else {
        errorMessage.textContent = '';
    }
});

document.getElementById('sign-in-form')?.addEventListener('submit', function(event) {
    event.preventDefault(); 

    var email = document.getElementById('email').value;
    var password = document.getElementById('password').value;
    var errorMessage = document.getElementById('sign-in-error');

    fetch('/api/check-credentials', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email: email, password: password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
          
            window.location.href = 'dashboard.html'; 
        } else {
  
            errorMessage.textContent = 'Incorrect email or password.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        errorMessage.textContent = 'An error occurred. Please try again.';
    });
});

document.getElementById('forgot-password-form')?.addEventListener('submit', function(event) {
    event.preventDefault();

    var email = document.getElementById('reset-email').value;
    var resetMessage = document.getElementById('reset-message');
    var resetError = document.getElementById('reset-error');

    fetch('/api/send-reset-link', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email: email })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {

            resetMessage.textContent = 'Reset link sent! Check your email.';
            resetError.textContent = '';
        } else {
 
            resetError.textContent = 'Error sending reset link. Please try again.';
            resetMessage.textContent = '';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resetError.textContent = 'An error occurred. Please try again.';
        resetMessage.textContent = '';
    });

    document.getElementById('signin-form').addEventListener('submit', function(e) {
        e.preventDefault(); 
    
        window.location.href = 'landing-page.html'; 
    });
});

document.getElementById('backButton').addEventListener('click', function () {
    window.location.href = 'landing-page.html'; 
});

document.getElementById('searchBtn').addEventListener('click', function () {
    const searchValue = document.getElementById('search').value;
    alert(`You searched for ID: ${searchValue}`);
});
