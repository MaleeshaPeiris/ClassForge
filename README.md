Steps to Run the App:

Clone the project

git clone https://github.com/MaleeshaPeiris/ClassForge.git

cd to classforge folder

run command -
docker build -t student-allocator .

After build complete run command -
docker run -p 5050:5000 student-allocator

To open the web app, in your browser, go to:
http://localhost:5050



Your input file should have these column headers:
gender_code,well-being,SES,achievement,psychological_distress,student_id
(will share the code to generate a sample data file shared)


Upload CSV file

Train model live

Allocate students into classes

View allocation results on the webpage