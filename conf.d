upstream ws-streamlit {
server 172.31.54.199:8501;
}
server {
    listen 80; 
    server_name 172.31.54.199;
location / {
proxy_pass http://ws-streamlit;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header Host $http_host;
      proxy_redirect off;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
    }
  }