document.addEventListener("DOMContentLoaded", function () {
  const form = document.querySelector(".php-email-form");
  const submitBtn = document.getElementById("submitBtn");
  const spinner = document.getElementById("spinner");
  const result = document.getElementById("result");

  form.addEventListener("submit", async function (e) {
      e.preventDefault();
      submitBtn.disabled = true;
      spinner.style.display = "inline-block";
      submitBtn.style.display = "none";

      const formData = new FormData(form);
      const response = await fetch("/contact", {
          method: "POST",
          body: formData,
      });

      const responseData = await response.json();
      if (responseData.status === "success") {
          result.innerHTML = responseData.message; 
          result.classList.add("text-success");
          result.classList.remove("text-danger");
      } else {
          result.innerHTML = responseData.message;
          result.classList.add("text-danger");
          result.classList.remove("text-success");
      }
      submitBtn.disabled = false;
      spinner.style.display = "none";
      submitBtn.style.display = "inline-block";
  });
});
