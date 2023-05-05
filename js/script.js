

fetch("stud.json")
  .then(function(response) {
    return response.json();
  })
  .then(function(stud) {
    let placeholder = document.querySelector("#data-out");
    let out = "";
    for (let student of stud) {
      out += `
        <tr>
          <td>${student.Name}</td>
          <td>${student.USN}</td>
          <td>${student.Class}</td>
          <td>${student['Previous Year Rank']}</td>
          <td><button id="myBtn">View</button></td>
          <td><input class="star" type="checkbox" title="bookmark page"></td>
        </tr>
      `;
    }

    placeholder.innerHTML = out;
  });
