FROM node:22.12.0-alpine3.20 AS build

WORKDIR /workspace/ADSMOD/client

COPY ADSMOD/client/package.json ADSMOD/client/package-lock.json ./
RUN npm ci

COPY ADSMOD/client ./
COPY ADSMOD/settings ../settings

RUN npm run build

FROM nginx:1.27.4-alpine

COPY docker/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY --from=build /workspace/ADSMOD/client/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
