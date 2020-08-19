FROM node:12-slim

WORKDIR /reactapp

COPY . .

RUN npm ci
RUN npm run build
RUN npm install -g serve