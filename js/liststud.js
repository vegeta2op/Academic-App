fetch("stud.json")
.then(function(response){
   return response.json();
})
.then(function(stud){
   let placeholder = document.querySelector("#data-out");
   let out = "";
   for(let student of stud){
      out += `
          <div class="student-info">
 <div class="grid-container">
  <div class="box">
    <span class="label">${student.Name}:</span>
    <span class="value">${student.USN}:</span>
  </div>

</div>
</div>
      `;
   }


   placeholder.innerHTML = out;

});
