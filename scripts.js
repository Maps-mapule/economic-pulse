// Password Validation for Sign-Up
document.getElementById('signup-form')?.addEventListener('submit', function(event) {
    var password = document.getElementById('password').value;
    var confirmPassword = document.getElementById('confirm-password').value;
    var errorMessage = document.getElementById('error-message');
    
    if (password !== confirmPassword) {
        errorMessage.textContent = 'Passwords do not match.';
        event.preventDefault(); // Prevent form submission
    } else {
        errorMessage.textContent = '';
        // Form is valid, it will be submitted
    }
});

// Incorrect Password Handling for Sign-In
document.getElementById('sign-in-form')?.addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission for AJAX handling

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
            // Handle successful login
            window.location.href = 'dashboard.html'; // Redirect to dashboard or another page
        } else {
            // Handle incorrect password
            errorMessage.textContent = 'Incorrect email or password.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        errorMessage.textContent = 'An error occurred. Please try again.';
    });
});

// Sending a Reset Password Link
document.getElementById('forgot-password-form')?.addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission for AJAX handling

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
            // Inform the user that the reset link has been sent
            resetMessage.textContent = 'Reset link sent! Check your email.';
            resetError.textContent = '';
        } else {
            // Inform the user of an error
            resetError.textContent = 'Error sending reset link. Please try again.';
            resetMessage.textContent = '';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resetError.textContent = 'An error occurred. Please try again.';
        resetMessage.textContent = '';
    });
});
