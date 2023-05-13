function verifyCredentials() {
  var email = document.getElementById("Email").value;
  var password = document.getElementById("Password").value;

  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/verify_credentials");
  xhr.setRequestHeader("Content-Type", "application/json");
   console.log("verifyCredentials function called");

  xhr.onreadystatechange = function() {
    if (xhr.readyState === XMLHttpRequest.DONE) {
      if (xhr.status === 200) {
        var response = JSON.parse(xhr.responseText);
        if (response.valid) {
          // Credentials are valid, perform the desired action (e.g., redirect to a dashboard page)
          window.location.href = "/dashboard";
        } else {
          // Credentials are not valid, display an error message
          document.getElementById("error-message").innerText = "Invalid email or password";
        }
      } else {
        // Handle any errors
        console.error(xhr.status);
      }
    }
  };

  // Send the email and password as JSON data to the Flask app
  var data = JSON.stringify({ "email": email, "password": password });
  xhr.send(data);
}
