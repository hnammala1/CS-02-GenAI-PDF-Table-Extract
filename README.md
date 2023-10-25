# CS-02-GenAI-PDF-Table-Extract

Steps:
1. Download the streamlit_deploy.yaml file and create a new stack in AWS cloudformation console.
2. Please make sure to use your PC's public IP address in the CIDR input parameter (e.g. 73.13.100.24/32) 
3. Once the CFN template is successfully running, please login to the EC2 instance created by the CFN
4. Please refer to the Stack output to find the instance id
5. Use SSM to login to the instance and open the EC2 terminal
6. Login as root and check the /var/log/cloud-init-output.log to see if the initialization scripts have successfully run
7. Login as ec2-user and navigate to /home/ce2-user/ directory
8. Run ./start_application.py to run the streamlit application
9. Find the Streamlit URL in the nohup.out and click on the External URL
