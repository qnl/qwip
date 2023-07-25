<template>
  <v-app>
    <nav-bar />
    <content-view />
  </v-app>
</template>

<script lang="ts" setup>
  import ContentView from './View.vue'
  import NavBar from './Navigation.vue'
</script>
