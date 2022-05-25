sudo docker run -it \

--mount type=bind,source=$(pwd)/python_code,target=/predict_churn \

-p 3000:5000 \

my_first_image_py:latest