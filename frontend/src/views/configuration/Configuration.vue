<template>
    <SideBar />
    <v-container fluid class="fill-height pa-0 flex-column">
        <v-row no-gutters style="width: 100%;">
            <v-col cols="12" class="flex-grow-1">
                <Folders />
            </v-col>
        </v-row>
        <v-row no-gutters style="width: 100%;">
            <v-col cols="12" class="flex-grow-1">
                <Parameters />
            </v-col>
        </v-row>
  </v-container>
</template>

<script lang="ts" setup>
    import SideBar from '@/components/navigation/Sidebar.vue'
    import Folders from './Folders.vue'
    import Parameters from './Parameters.vue'
</script>