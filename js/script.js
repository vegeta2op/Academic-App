fetch("stud.json")
  .then(function(response) {
    return response.json();
  })
  .then(function(stud) {
    let placeholder = document.querySelector("#data-out");
    let out = "";
    for (let student of stud) {
      let starred = student['Starred']  === 'Y' ? '' : 'checked';
      out += `
        <tr>
          <td>${student.Name}</td>
          <td>${student.USN}</td>
          <td>${student.Class}</td>
          <td>${student['Previous Year Rank']}</td>
          <td><button id="myBtn">View</button></td>
          <td><input class="star" type="checkbox" title="bookmark page" ${starred} data-id="${student.USN}"></td>
        </tr>
      `;
    }

    placeholder.innerHTML = out;

let starInputs = document.querySelectorAll(".star");
for (let studentIndex = 0; studentIndex < stud.length; studentIndex++) {
  starInputs[studentIndex].addEventListener("change", function(event) {
    let studentUSN = event.target.getAttribute('data-id');
    let studentIndex = stud.findIndex(student => student.USN === studentUSN);
    let newValue = event.target.checked ? 'N' : 'Y'; // swap the values
    stud[studentIndex]['Starred'] = newValue;
    let json = JSON.stringify(stud);
    fetch('stud.json', {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json'
      },
      body: json
    }).then(function(response) {
      console.log('Success:', JSON.stringify(response));
    }).catch(function(error) {
      console.error('Error:', error);
    });
  });
}

  });
